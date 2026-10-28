"""ConvLSTM forecasters (model A and the larger-reach model D) and the symmetrised wrapper."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import G, apply_g, apply_g_inv

N_IN, N_OUT, N_STEPS = 5, 5, 9


class ConvLSTMCell(nn.Module):
    """[i, f, o, g] = Conv([u, h_prev]);  c = s(f)*c_prev + s(i)*tanh(g);  h = s(o)*tanh(c).

    No peephole terms. One convolution produces all four gates.
    """

    def __init__(self, in_ch, hid_ch, kernel=3, dilation=1):
        super().__init__()
        self.hid_ch = hid_ch
        pad = dilation * (kernel - 1) // 2  # keeps the 16x16 grid ("same" zero padding)
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=pad,
                              dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """pixel_unshuffle(4) -> stacked ConvLSTM layers -> 1x1 conv -> pixel_shuffle(4) -> sigmoid.

    layers: list of (hidden_channels, kernel, dilation), one entry per ConvLSTM layer.
    Model A is layers=[(64, 3, 1), (64, 3, 1)].
    """

    def __init__(self, layers=((64, 3, 1), (64, 3, 1)), patch=4):
        super().__init__()
        self.patch = patch
        self.layer_cfg = [tuple(l) for l in layers]
        in_ch = patch * patch
        cells = []
        for hid, k, d in self.layer_cfg:
            cells.append(ConvLSTMCell(in_ch, hid, k, d))
            in_ch = hid
        self.cells = nn.ModuleList(cells)
        self.head = nn.Conv2d(in_ch, patch * patch, 1, bias=True)

    def init_state(self, B, H, W, device, dtype):
        return [(torch.zeros(B, c.hid_ch, H, W, device=device, dtype=dtype),) * 2 for c in self.cells]

    def step(self, frame, states):
        u = F.pixel_unshuffle(frame, self.patch)
        new_states = []
        for cell, st in zip(self.cells, states):
            h, c = cell(u, st)
            new_states.append((h, c))
            u = h
        return torch.sigmoid(F.pixel_shuffle(self.head(u), self.patch)), new_states

    def forward(self, X):
        """X: (B, 5, 1, 64, 64) in [0, 1] -> Y_hat: (B, 5, 1, 64, 64).

        9-step rollout without teacher forcing: steps 1-5 read the input frames,
        steps 6-9 read the network's own previous prediction (gradients flow through it).
        Outputs of steps 5-9 form Y_hat.
        """
        B, _, _, H, W = X.shape
        states = self.init_state(B, H // self.patch, W // self.patch, X.device, X.dtype)
        outs, pred = [], None
        for tau in range(1, N_STEPS + 1):
            u = X[:, tau - 1] if tau <= N_IN else pred
            pred, states = self.step(u, states)
            if tau >= N_IN:
                outs.append(pred)
        return torch.stack(outs, dim=1)


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^{-1}(f(g(X))) over the 8 symmetries of the square."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, X):
        B = X.shape[0]
        # all eight transformed copies go through f as one batch of 8B videos
        Xg = torch.cat([apply_g(X, g) for g in G], dim=0)
        Yg = self.f(Xg)
        out = sum(apply_g_inv(Yg[i * B:(i + 1) * B], g) for i, g in enumerate(G))
        return out / len(G)


def model_A():
    return ConvLSTMForecaster([(64, 3, 1), (64, 3, 1)])


# Candidate larger-reach designs for model D (compared on the validation split only).
D_CANDIDATES = {
    # same parameter count as A; layer 2 uses a dilated 3x3 kernel
    "dil12": dict(layers=[(64, 3, 1), (64, 3, 2)]),
    # 5x5 kernels in both layers, hidden width reduced to stay under the 1.5x budget
    "k55h46": dict(layers=[(46, 5, 1), (46, 5, 1)]),
    # bigger patch: fold 8x8 patches, so one grid cell is 8 original pixels
    "p8": dict(layers=[(64, 3, 1), (64, 3, 1)], patch=8),
}


def build(name):
    if name == "A":
        return model_A()
    return ConvLSTMForecaster(**D_CANDIDATES[name])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
