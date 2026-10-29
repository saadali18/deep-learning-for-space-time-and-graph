"""Load trained models / reference predictors and compute (and cache) their test metrics."""
import json
import os

import numpy as np
import torch

from .data import load_videos, split
from .evaluate import (TRANSFORMS, Blank, CopyLast, consistency, get_device, inference_time, predict,
                       scores)
from .models import Symmetrised, build
from .style import RES_DIR
from .train import CKPT_DIR

SEEDS = (0, 1)
_VIDEOS = {}


def test_videos():
    if "test" not in _VIDEOS:
        _VIDEOS["test"] = split(load_videos(), "test")
    return _VIDEOS["test"]


def load_model(name, seed, device=None):
    """name: a trained run ('A', 'B', 'C', 'D', 'DB') or 'sym:<run>' for its symmetrised version."""
    device = device or get_device()
    if name == "Blank":
        return Blank().to(device)
    if name == "Copy-last":
        return CopyLast().to(device)
    sym = name.startswith("sym:")
    run = name[4:] if sym else name
    ck = torch.load(os.path.join(CKPT_DIR, f"{run}_s{seed}.pt"), map_location="cpu")
    m = build(ck["arch"])
    m.load_state_dict(ck["state_dict"])
    m = m.to(device).eval()
    return Symmetrised(m).eval() if sym else m


def train_log(name, seed):
    run = name[4:] if name.startswith("sym:") else name
    with open(os.path.join(CKPT_DIR, f"{run}_s{seed}.json")) as fh:
        return json.load(fh)


def horizon_metrics(model, videos, device):
    """Table 2 quantities: MSE_1..5 and E(5) at 1x, t = 1."""
    P, Y, M = predict(model, videos, 1, 1, device)
    return scores(P, Y, M)


def full_metrics(model, videos, device, per_video_path=None):
    """Every quantity of Tables 3-6 for one model."""
    out = {}
    P, Y, M = predict(model, videos, 1, 1, device)
    out.update({k + "_1x" if k in ("E5", "E2") else k: v for k, v in scores(P, Y, M).items()})
    per_video = {}
    for tn, T in TRANSFORMS.items():
        C, E, _ = consistency(model, videos, T, device)
        out[f"C{tn}"], out[f"E{tn}"] = float(C.mean()), float(E.mean())
        per_video[f"C{tn}"], per_video[f"E{tn}"] = C.numpy(), E.numpy()
    for k in (2, 3):
        P, Y, M = predict(model, videos, 1, k, device)
        sc = scores(P, Y, M)
        out[f"E2_{k}x"] = sc["E2"]
        if "E5" in sc:
            out[f"E5_{k}x"] = sc["E5"]
    if per_video_path:
        np.savez(per_video_path, **per_video)
    return out


def cached(kind, name, seed, compute):
    """Cache a metrics dict in results/<kind>_<name>_s<seed>.json."""
    path = os.path.join(RES_DIR, f"{kind}_{name.replace(':', '-')}_s{seed}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    out = compute()
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def metrics_for(name, seed=0, kind="full"):
    """Cached metrics (+ inference time and parameters/training time for learned models)."""
    dev = get_device()

    def compute():
        model = load_model(name, seed, dev)
        vids = test_videos()
        if kind == "horizon":
            return horizon_metrics(model, vids, dev)
        pv = os.path.join(RES_DIR, f"pervideo_{name.replace(':', '-')}_s{seed}.npz")
        out = full_metrics(model, vids, dev, pv)
        out["inference_s"] = inference_time(model, vids, dev, repeats=3)
        if name not in ("Blank", "Copy-last"):
            log = train_log(name, seed)
            out["params"] = log["params"]
            out["train_s_per_epoch"] = log["train_seconds_per_epoch"]
        return out

    return cached(kind, name, seed, compute)


def per_video(name, seed=0):
    return dict(np.load(os.path.join(RES_DIR, f"pervideo_{name.replace(':', '-')}_s{seed}.npz")))
