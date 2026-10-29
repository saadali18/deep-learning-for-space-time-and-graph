"""Prediction, error metrics, consistency gaps and inference timing."""
import time

import torch

from .data import batch_window, mirror, rotate, to_float


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


class Blank(torch.nn.Module):
    def forward(self, X):
        return torch.zeros_like(X)


class CopyLast(torch.nn.Module):
    def forward(self, X):
        return X[:, -1:].expand_as(X).clone()


TRANSFORMS = {"M": mirror, "R": lambda x: rotate(x, 1)}


@torch.no_grad()
def predict(model, videos, t=1, k=1, device=None, batch=32, transform=None):
    """Forecasts for uint8 videos (N, 20, 1, 64, 64) at start t and speed k.

    transform (callable) is applied to every frame of the input and the true future.
    Returns predictions, targets (float, CPU, (N, 5, 1, 64, 64)) and the target mask (N, 5).
    """
    device = device or get_device()
    model.eval()
    preds, targets, masks = [], [], []
    for s in range(0, videos.shape[0], batch):
        X, Y, mask = batch_window(videos[s:s + batch], t, k)
        X, Y = to_float(X.to(device)), to_float(Y)
        if transform is not None:
            X, Y = transform(X), transform(Y)
        preds.append(model(X).cpu())
        targets.append(Y)
        masks.append(mask.cpu())
    return torch.cat(preds), torch.cat(targets), torch.cat(masks)


def per_video_horizon_mse(P, Y):
    """MSE_h per video: mean over the 64x64 pixels of each forecast horizon -> (N, 5)."""
    return (P - Y).pow(2).mean(dim=(2, 3, 4))


def scores(P, Y, mask):
    """MSE_1..5 (only where every video has the frame), E(5) and E(2), averaged over videos."""
    mse = per_video_horizon_mse(P, Y)
    out = {}
    for h in range(5):
        if mask[:, h].all():
            out[f"MSE{h + 1}"] = float(mse[:, h].mean())
    if mask.all():
        out["E5"] = float(mse.mean(1).mean())
    out["E2"] = float(mse[:, :2].mean(1).mean())
    return out


@torch.no_grad()
def consistency(model, videos, T, device=None):
    """Per-video C_T = MSE(f(T X), T f(X)) and E_T = MSE(f(T X), T Y) at 1x, t=1.

    Returns (C, E, extras) where C and E are (N,) per-video values averaged over pixels and five horizons.
    """
    P, Y, _ = predict(model, videos, 1, 1, device)
    PT, YT, _ = predict(model, videos, 1, 1, device, transform=T)
    TP = T(P)
    C = per_video_horizon_mse(PT, TP).mean(1)
    E = per_video_horizon_mse(PT, YT).mean(1)
    return C, E, dict(P=P, Y=Y, PT=PT, YT=YT, TP=TP)


@torch.no_grad()
def inference_time(model, videos, device=None, batch=32, repeats=1):
    """Wall-clock time to predict all videos (1x, t=1), counting only after each batch is on the device."""
    device = device or get_device()
    model.eval()
    batches = []
    for s in range(0, videos.shape[0], batch):
        X, _, _ = batch_window(videos[s:s + batch], 1, 1)
        batches.append(to_float(X.to(device)))
    model(batches[0])  # warm-up (kernel compilation), not timed
    device_sync(device)
    best = float("inf")
    for _ in range(repeats):
        total = 0.0
        for X in batches:
            device_sync(device)
            t0 = time.perf_counter()
            model(X)
            device_sync(device)
            total += time.perf_counter() - t0
        best = min(best, total)
    return best
