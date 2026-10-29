"""Train one forecaster under the shared protocol.

Adam(lr=1e-3), no schedule, batch 32, 30 epochs, final-epoch weights kept.
One epoch = one pass over the 8,000 training videos in shuffled order with one random window per video.
The seed controls initialisation, shuffling, window sampling and augmentation sampling.

usage: python -m mm.train --name A --arch A --seed 0
       python -m mm.train --name B --arch A --sym-aug --seed 0
       python -m mm.train --name C --arch A --speed-aug --seed 0
"""
import argparse
import json
import os
import time

import torch

from .data import G, allowed_starts, apply_g, batch_window, load_videos, split, to_float
from .evaluate import device_sync, get_device, predict, scores
from .models import build, count_params

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(HERE, "checkpoints")


def masked_video_mse(pred, Y, mask):
    """Squared error averaged over pixels and available target frames of each video, then over videos."""
    se = (pred - Y).pow(2).mean(dim=(2, 3, 4))  # (B, 5): per-frame pixel mean
    m = mask.float()
    return ((se * m).sum(1) / m.sum(1)).mean()


def sample_windows(B, speed_aug, gen):
    """Per-video speed k and start t following the rules in Setup."""
    if speed_aug:
        k = torch.randint(1, 4, (B,), generator=gen)
        t = torch.empty(B, dtype=torch.long)
        for kk in (1, 2, 3):
            sel = k == kk
            starts = torch.tensor(allowed_starts(kk, True))
            t[sel] = starts[torch.randint(len(starts), (int(sel.sum()),), generator=gen)]
    else:
        k = torch.ones(B, dtype=torch.long)
        starts = torch.tensor(allowed_starts(1, False))
        t = starts[torch.randint(len(starts), (B,), generator=gen)]
    return t, k


def apply_per_video(x, gidx):
    """Apply symmetry G[gidx[b]] to video b (the same g is used for input and target)."""
    out = torch.empty_like(x)
    for i, g in enumerate(G):
        sel = (gidx == i).to(x.device)
        if sel.any():
            out[sel] = apply_g(x[sel], g)
    return out


def train(name, arch, seed, speed_aug=False, sym_aug=False, epochs=30, batch=32, lr=1e-3,
          eval_every=5, out_dir=CKPT_DIR):
    dev = get_device()
    torch.manual_seed(seed)  # initialisation
    gen = torch.Generator().manual_seed(seed)  # shuffling, windows, augmentation
    model = build(arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    videos = load_videos()
    train_v = split(videos, "train").to(dev)
    val_v = split(videos, "val")
    n = train_v.shape[0]

    log = dict(name=name, arch=arch, seed=seed, speed_aug=speed_aug, sym_aug=sym_aug,
               params=count_params(model), device=str(dev), epochs=[])
    print(f"[{name} s{seed}] arch={arch} params={log['params']} device={dev}", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        device_sync(dev)
        t0 = time.perf_counter()
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            B = len(idx)
            t, k = sample_windows(B, speed_aug, gen)
            X, Y, mask = batch_window(train_v[idx.to(dev)], t, k)
            X, Y = to_float(X), to_float(Y)
            if sym_aug:
                gidx = torch.randint(len(G), (B,), generator=gen)
                X, Y = apply_per_video(X, gidx), apply_per_video(Y, gidx)
            loss = masked_video_mse(model(X), Y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * B
            cnt += B
        device_sync(dev)
        ep_time = time.perf_counter() - t0
        rec = dict(epoch=ep, train_loss=float(tot / cnt), seconds=ep_time)
        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            for kk in (1, 2, 3):
                P, Yv, Mv = predict(model, val_v, t=1, k=kk, device=dev)
                sc = scores(P, Yv, Mv)
                rec[f"val_E2_{kk}x"] = sc["E2"]
                if kk == 1:
                    rec["val_E5_1x"] = sc["E5"]
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)

    log["train_seconds_per_epoch"] = sum(r["seconds"] for r in log["epochs"]) / epochs
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), arch=arch, log=log),
               os.path.join(out_dir, f"{name}_s{seed}.pt"))
    with open(os.path.join(out_dir, f"{name}_s{seed}.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    return model, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-aug", action="store_true")
    ap.add_argument("--sym-aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-dir", default=CKPT_DIR)
    a = ap.parse_args()
    train(a.name, a.arch, a.seed, a.speed_aug, a.sym_aug, a.epochs, out_dir=a.out_dir)
