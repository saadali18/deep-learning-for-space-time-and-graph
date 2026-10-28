"""Supplementary check for Q2.1 (not part of Figure 1): direction-histogram symmetry on 1,000 training videos."""
import json
import os

import numpy as np

from mm.data import load_videos, split
from mm.style import RES_DIR
from scripts.figure1_flow import pooled_flow


def hist(th):
    return np.histogram(th % 360.0, bins=18, range=(0, 360))[0] / len(th)


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def main():
    dx, dy = pooled_flow(split(load_videos(), "train")[:1000])
    th = np.degrees(np.arctan2(-dy, dx)) % 360.0
    h = hist(th)
    out = dict(n_videos=1000, n_pixels=int(len(th)), median_speed=float(np.median(np.hypot(dx, dy))),
               dir_frac=h.tolist(), dir_min=float(h.min()), dir_max=float(h.max()),
               tv_uniform=tv(h, np.full(18, 1 / 18)), tv_mirror=tv(h, hist(180 - th)),
               tv_rot90=tv(h, hist(th + 90)), tv_rot180=tv(h, hist(th + 180)))
    print(json.dumps(out, indent=1))
    with open(os.path.join(RES_DIR, "flow_supplement_1000.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
