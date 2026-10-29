"""Figure builders shared by the baseline (model A) and final-model experiments."""
import json
import os

import numpy as np
import torch

from mm.data import batch_window, to_float
from mm.evaluate import TRANSFORMS, consistency, get_device, predict
from mm.style import GIF_DIR, INK2, RES_DIR, SERIES, plt, save
from mm.viz import grid, n_blobs, overlap_profile, save_gif

SEL_PATH = os.path.join(RES_DIR, "figure_videos.json")
T_NAMES = {"M": "mirror", "R": "rotation"}


def selections():
    with open(SEL_PATH) as fh:
        return json.load(fh)


def store_selection(key, value):
    sel = selections() if os.path.exists(SEL_PATH) else {}
    sel[key] = value
    with open(SEL_PATH, "w") as fh:
        json.dump(sel, fh, indent=1)


def frames(x):
    return [f[0].numpy() for f in x]


def fig3_panel(model, videos, idx, tn, tag, title):
    """One Figure 3 panel (four rows of ten frames) plus its GIF. Returns the GIF path."""
    T = TRANSFORMS[tn]
    v = videos[idx:idx + 1]
    C, E, d = consistency(model, v, T, get_device())
    X, _, _ = batch_window(v, 1, 1)
    X = to_float(X)[0]
    TX = T(X)
    rows = [frames(X) + frames(d["P"][0]),
            frames(TX) + frames(d["YT"][0]),
            frames(TX) + frames(d["PT"][0]),
            frames(TX) + frames(d["TP"][0])]
    labels = ["(1) X | f(X)", "(2) T(X) | T(Y)", "(3) T(X) | f(T(X))", "(4) T(X) | T(f(X))"]
    labels = [l.replace("T(", f"{tn}(") for l in labels]
    cols = [f"in {i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
    fig = grid(rows, labels, cols, sep_after=5,
               suptitle=f"{title}: test video {9000 + idx}, T = {tn} ({T_NAMES[tn]}); per-video C_{tn} = {C.item():.3g}, E_{tn} = {E.item():.3g}")
    save(fig, f"{tag}_{tn}_v{9000 + idx}.png")
    gif = save_gif(rows, labels, os.path.join(GIF_DIR, f"{tag}_{tn}_v{9000 + idx}.gif"))
    return gif, float(C), float(E)


def fig5(model, videos, idx, name, title):
    v = videos[idx:idx + 1]
    rows, labels = [], []
    for k in (1, 2, 3):
        P, Y, _ = predict(model, v, 1, k, get_device())
        X, _, _ = batch_window(v, 1, k)
        X = to_float(X)[0]
        rows.append(frames(X) + [P[0, 0, 0].numpy(), P[0, 1, 0].numpy(), Y[0, 0, 0].numpy(), Y[0, 1, 0].numpy()])
        labels.append(f"{k}×")
    cols = [f"in {i}" for i in range(1, 6)] + ["pred h=1", "pred h=2", "true h=1", "true h=2"]
    fig = grid(rows, labels, cols, sep_after=5, size=0.85, suptitle=f"{title}: test video {9000 + idx} at 1×, 2×, 3× (t = 1)")
    return save(fig, name)


def pick_fig5_video(videos):
    """First test video whose digits do not overlap in any displayed true frame at 1x, 2x and 3x."""
    shown = set()
    for k in (1, 2, 3):
        shown.update(1 + k * i for i in range(7))  # five inputs and the first two targets
    shown = sorted(f - 1 for f in shown if f <= 20)
    for i in range(videos.shape[0]):
        # exactly two well-separated blobs (the two digits) in every displayed frame
        if all(n_blobs(videos[i, f, 0].numpy()) == 2 for f in shown):
            return i
    raise RuntimeError("no non-overlapping video")


def fig4(pv, marks, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, tn in zip(axes, ("M", "R")):
        c = pv[f"C{tn}"]
        bins = np.logspace(np.log10(max(c.min(), 1e-6)), np.log10(c.max()), 40)
        h, e = np.histogram(c, bins=bins)
        ax.bar(e[:-1], h, width=np.diff(e), align="edge", color=SERIES[0], edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
        ax.axvline(c.mean(), color=INK2, lw=1, ls="--")
        ax.text(c.mean(), ax.get_ylim()[1] * 0.97, " mean", fontsize=7, color=INK2, va="top")
        for j, (idx, lab) in enumerate(marks[tn]):
            ax.axvline(c[idx], color=SERIES[1], lw=1.6)
            ax.text(c[idx], ax.get_ylim()[1] * (0.82 - 0.14 * j), f" {lab}", fontsize=7, color=SERIES[1], va="top",
                    ha="right" if j == 0 else "left")
        ax.set_xlabel(f"per-video C_{tn} (log scale)")
        ax.set_ylabel("number of test videos")
        ax.set_title(f"({'a' if tn == 'M' else 'b'}) {T_NAMES[tn]}")
    fig.suptitle(title, fontsize=9.5)
    return save(fig, name)


def fig2(model, videos, idxs, name):
    rows, labels = [], []
    for i in idxs:
        v = videos[i:i + 1]
        P, Y, _ = predict(model, v, 1, 1, get_device())
        X, _, _ = batch_window(v, 1, 1)
        X = to_float(X)[0]
        rows += [frames(X), frames(P[0]), frames(Y[0])]
        labels += [f"video {9000 + i}\ninput (1–5)", "prediction (6–10)", "truth (6–10)"]
    fig = grid(rows, labels, size=0.8)
    return save(fig, name)


def pick_fig2_videos(videos):
    prof = np.stack([overlap_profile(videos[i]) for i in range(videos.shape[0])])
    clean = [i for i in range(len(prof)) if prof[i, :10].max() < 0.02][:2]
    overl = [i for i in range(len(prof)) if prof[i, 5:10].max() > 0.2][0]  # strong overlap in the target frames
    return clean + [overl]


def torch_seed():
    torch.manual_seed(0)
