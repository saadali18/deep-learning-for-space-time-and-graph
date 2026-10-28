"""Figure 1: direction and speed of dense optical flow on the first 50 training videos."""
import json
import os

import cv2
import numpy as np

from mm.data import load_videos, split
from mm.style import INK, INK2, RES_DIR, SERIES, plt, save


def pooled_flow(videos):
    dxs, dys = [], []
    for v in videos:
        frames = v[:, 0].numpy()  # uint8 frames (20, 64, 64): already intensity*255
        for s in range(19):
            prev, nxt = frames[s], frames[s + 1]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            mag = np.hypot(dx, dy)
            # intensity > 0.2 on the [0,1] scale  <=>  uint8 value > 51 (0.2*255)
            keep = (prev / 255.0 > 0.2) & (nxt / 255.0 > 0.2) & (mag > 0.5)
            dxs.append(dx[keep])
            dys.append(dy[keep])
    return np.concatenate(dxs), np.concatenate(dys)


def main():
    videos = split(load_videos(), "train")[:50]
    assert videos.dtype.is_floating_point is False  # stored as uint8 = float frame * 255
    dx, dy = pooled_flow(videos)
    theta = np.degrees(np.arctan2(-dy, dx)) % 360.0
    speed = np.hypot(dx, dy)
    n = len(speed)
    med = float(np.median(speed))

    dir_counts, edges = np.histogram(theta, bins=18, range=(0, 360))
    dir_frac = dir_counts / n
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    ax.bar(edges[:-1], dir_frac, width=20, align="edge", color=SERIES[0], edgecolor="white", linewidth=1)
    ax.axhline(1 / 18, color=INK, lw=1.2, ls="--")
    ax.text(362, 1 / 18, " uniform\n 1/18", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlim(0, 360)
    ax.set_xlabel("flow direction θ (degrees; 0° = right, 90° = up)")
    ax.set_ylabel("fraction of kept pixels")
    ax.set_title("(a) direction")

    ax = axes[1]
    bins = np.arange(0.5, np.ceil(speed.max()) + 0.25, 0.25)
    sc, se = np.histogram(speed, bins=bins)
    ax.bar(se[:-1], sc / n, width=0.25, align="edge", color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax.axvline(med, color=SERIES[1], lw=2)
    ax.text(med + 0.15, ax.get_ylim()[1] * 0.92, f"median {med:.2f} px/frame", color=INK, fontsize=8)
    ax.set_xlabel("flow speed (pixels per frame)")
    ax.set_ylabel("fraction of kept pixels")
    ax.set_title("(b) speed")
    ax.set_xlim(0.5, min(12, speed.max() + 0.5))
    save(fig, "fig1_flow.png")

    # summary statistics used in the answers to Q2.1
    quad = {q: float(((theta >= lo) & (theta < lo + 90)).mean())
            for q, lo in (("0-90", 0), ("90-180", 90), ("180-270", 180), ("270-360", 270))}
    stats = dict(n_pixels=int(n), n_pairs=50 * 19, median_speed=med,
                 mean_speed=float(speed.mean()),
                 speed_percentiles={p: float(np.percentile(speed, p)) for p in (10, 25, 75, 90, 95, 99)},
                 dir_frac=dir_frac.tolist(), dir_min=float(dir_frac.min()), dir_max=float(dir_frac.max()),
                 quadrant_frac=quad,
                 mean_dx=float(dx.mean()), mean_dy=float(dy.mean()),
                 frac_right=float((dx > 0).mean()), frac_up=float((dy < 0).mean()))
    print(json.dumps(stats, indent=1))
    with open(os.path.join(RES_DIR, "figure1_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
