import math

import torch
from torch import nn
from torch.nn import functional as F

class SeriesDecomposition(nn.Module):
    """Centred moving average with replicate padding. Returns (seasonal, trend), both [B, L, C]."""

    def __init__(self, kernel):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):
        half = self.kernel // 2
        padded = F.pad(x.transpose(1, 2), (half, half), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


def delay_scores(queries, keys):
    """[B, heads, features, L] x2 -> [B, L]: circular correlation at every delay, via FFT."""
    queries = queries - queries.mean(-1, keepdim=True)
    keys = keys - keys.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(queries, dim=-1) * torch.fft.rfft(keys, dim=-1).conj()
    return torch.fft.irfft(spectrum, n=queries.shape[-1], dim=-1).mean((1, 2))


def aggregate_delays(values, delays, weights):
    """values [B, h, f, L]; delays, weights [B, K] -> sum_j w_j * values[(t - tau_j) mod L]."""
    batch, heads, features, length = values.shape
    time = torch.arange(length, device=values.device)
    source = (time[None, None, :] - delays[:, :, None]) % length          # [B, K, L]
    mixed = torch.zeros_like(values)
    for j in range(delays.shape[1]):
        index = source[:, j][:, None, None, :].expand(batch, heads, features, length)
        mixed = mixed + weights[:, j, None, None, None] * values.gather(-1, index)
    return mixed

class AutoCorrelation(nn.Module):
    """Auto-Correlation with time-delay aggregation (replaces dot-product attention)."""

    def __init__(self, d_model, heads, factor=1.0, min_delay=0):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads, self.factor, self.min_delay = heads, factor, min_delay
        self.query, self.key, self.value = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):              
        batch, length, width = queries.shape
        if keys.shape[1] > length:                         
            keys, values = keys[:, :length], values[:, :length]          # as in the paper
        elif keys.shape[1] < length:
            pad = length - keys.shape[1]
            keys, values = F.pad(keys, (0, 0, 0, pad)), F.pad(values, (0, 0, 0, pad))
        shape = (batch, length, self.heads, width // self.heads)
        q, k, v = (layer(x).view(shape).permute(0, 2, 3, 1)               # [B, h, d_h, L]
                   for layer, x in ((self.query, queries), (self.key, keys), (self.value, values)))
        scores = delay_scores(q, k)                                       # [B, L]
        band = scores[:, self.min_delay:]
        top_k = max(1, min(int(self.factor * math.log(length)), band.shape[-1]))
        selected, index = band.topk(top_k, dim=-1)
        weights = selected.softmax(-1)                                    # [B, K]
        mixed = aggregate_delays(v, index + self.min_delay, weights)      # [B, h, d_h, L]
        return self.out(mixed.permute(0, 3, 1, 2).reshape(batch, length, width))


class DataEmbedding(nn.Module):
    """Value embedding (Conv1d, kernel 3) + optional linear covariate embedding. No positions."""

    def __init__(self, c_in, c_cov, d_model, dropout):
        super().__init__()
        self.value = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        self.covariates = nn.Linear(c_cov, d_model, bias=False) if c_cov > 0 else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cov=None):                        # x [B, L, c_in], cov [B, L, c_cov]
        embedded = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.covariates is not None:
            embedded = embedded + self.covariates(cov)
        return self.drop(embedded)


class SeasonalLayerNorm(nn.Module):
    """The paper's layer norm for the seasonal part: LayerNorm, then remove the mean over time."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


def feed_forward(d_model, d_ff, dropout):
    return nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_ff, d_model, bias=False))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, kernel, factor, min_delay, dropout):
        super().__init__()
        self.autocorrelation = AutoCorrelation(d_model, heads, factor, min_delay)
        self.ffn = feed_forward(d_model, d_ff, dropout)
        self.decomp1, self.decomp2 = SeriesDecomposition(kernel), SeriesDecomposition(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.drop(self.autocorrelation(x, x, x)))   # keep seasonal only
        x, _ = self.decomp2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, kernel, factor, min_delay, dropout, c_out):
        super().__init__()
        self.self_autocorrelation = AutoCorrelation(d_model, heads, factor, min_delay)
        self.cross_autocorrelation = AutoCorrelation(d_model, heads, factor, min_delay)
        self.ffn = feed_forward(d_model, d_ff, dropout)
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomposition(kernel) for _ in range(3))
        self.trend_projection = nn.Conv1d(d_model, c_out, 3, padding=1, padding_mode="circular",
                                          bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, encoder_out):
        x, trend1 = self.decomp1(x + self.drop(self.self_autocorrelation(x, x, x)))
        x, trend2 = self.decomp2(
            x + self.drop(self.cross_autocorrelation(x, encoder_out, encoder_out)))
        x, trend3 = self.decomp3(x + self.drop(self.ffn(x)))
        residual_trend = self.trend_projection((trend1 + trend2 + trend3).transpose(1, 2))
        return x, residual_trend.transpose(1, 2)           # [B, L, d], [B, L, c_out]

class Autoformer(nn.Module):
    """Encoder-decoder Autoformer for a univariate target with optional covariates.

    forward(x, cov_enc, cov_dec):
        x        [B, context, 1]               scaled target history
        cov_enc  [B, context, c_cov]           covariates aligned with the history (or None)
        cov_dec  [B, label_len + horizon, c_cov] covariates aligned with the decoder input (or None)
        returns  [B, horizon]                  scaled forecast
    """

    def __init__(self, context=336, label_len=168, horizon=168, c_cov=0, d_model=32, heads=4,
                 d_ff=64, e_layers=1, d_layers=1, kernel=25, factor=1.0, min_delay=0,
                 dropout=0.05, trend_init="mean", cov_trend=False):
        super().__init__()
        self.context, self.label_len, self.horizon, self.c_cov = context, label_len, horizon, c_cov
        if trend_init not in ("mean", "learned"):
            raise ValueError("trend_init must be 'mean' (paper) or 'learned'")
        self.trend_init = trend_init
        if trend_init == "learned":
            self.trend_last = nn.Parameter(torch.zeros(horizon))
            self.trend_mean = nn.Parameter(torch.ones(horizon))
            self.trend_bias = nn.Parameter(torch.zeros(horizon))
        if cov_trend and c_cov == 0:
            raise ValueError("cov_trend needs covariates (c_cov > 0)")
        self.cov_trend = nn.Linear(c_cov, 1) if cov_trend else None
        self.decomp = SeriesDecomposition(kernel)
        self.enc_embedding = DataEmbedding(1, c_cov, d_model, dropout)
        self.dec_embedding = DataEmbedding(1, c_cov, d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, kernel, factor, min_delay, dropout)
            for _ in range(e_layers)])
        self.encoder_norm = SeasonalLayerNorm(d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, kernel, factor, min_delay, dropout, 1)
            for _ in range(d_layers)])
        self.decoder_norm = SeasonalLayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x, cov_enc=None, cov_dec=None):
        batch = x.shape[0]
        seasonal, trend = self.decomp(x)
        if self.trend_init == "mean":                                           
            future_trend = x.mean(dim=1, keepdim=True).expand(batch, self.horizon, 1)
        else:
            future_trend = (self.trend_last[None, :, None] * x[:, -1:, :]  + self.trend_mean[None, :, None] * x.mean(dim=1, keepdim=True) + self.trend_bias[None, :, None])                  # [B, H, 1]
        if self.cov_trend is not None:
            future_trend = future_trend + self.cov_trend(cov_dec[:, -self.horizon:, :])
        zeros = x.new_zeros(batch, self.horizon, 1)
        trend_init = torch.cat([trend[:, -self.label_len:], future_trend], dim=1)
        seasonal_init = torch.cat([seasonal[:, -self.label_len:], zeros], dim=1)

        encoded = self.enc_embedding(x, cov_enc)
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)

        decoded, trend_acc = self.dec_embedding(seasonal_init, cov_dec), trend_init
        for layer in self.decoder_layers:
            decoded, residual_trend = layer(decoded, encoded)
            trend_acc = trend_acc + residual_trend
        seasonal_out = self.projection(self.decoder_norm(decoded))
        return (seasonal_out + trend_acc)[:, -self.horizon:, 0]                  # [B, H]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)