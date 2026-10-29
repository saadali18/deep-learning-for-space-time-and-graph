"""Q2.1 supplement: Figure 1 statistics with and without frame pairs in which the digits overlap."""
import json
import os

import cv2
import numpy as np

from mm.data import load_videos, split
from mm.style import RES_DIR
from mm.viz import overlap_profile


def main():
    vids = split(load_videos(), "train")[:50]
    groups = {"all": ([], []), "no_overlap": ([], []), "overlap": ([], [])}
    n_pairs = {"no_overlap": 0, "overlap": 0}
    for v in vids:
        prof = overlap_profile(v)
        fr = v[:, 0].numpy()
        for s in range(19):
            flow = cv2.calcOpticalFlowFarneback(fr[s], fr[s + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            keep = (fr[s] / 255.0 > 0.2) & (fr[s + 1] / 255.0 > 0.2) & (np.hypot(dx, dy) > 0.5)
            g = "overlap" if max(prof[s], prof[s + 1]) > 0.02 else "no_overlap"
            n_pairs[g] += 1
            for k in ("all", g):
                groups[k][0].append(dx[keep])
                groups[k][1].append(dy[keep])
    out = dict(n_pairs=n_pairs)
    for k, (dxs, dys) in groups.items():
        dx, dy = np.concatenate(dxs), np.concatenate(dys)
        th = np.degrees(np.arctan2(-dy, dx)) % 360
        h = np.histogram(th, bins=18, range=(0, 360))[0] / len(th)
        hm = np.histogram((180 - th) % 360, bins=18, range=(0, 360))[0] / len(th)
        hr = np.histogram((th + 90) % 360, bins=18, range=(0, 360))[0] / len(th)
        sp = np.hypot(dx, dy)
        out[k] = dict(n_pixels=int(len(sp)), median_speed=float(np.median(sp)), frac_below_2px=float((sp < 2).mean()),
                      dir_min=float(h.min()), dir_max=float(h.max()),
                      tv_mirror=0.5 * float(np.abs(h - hm).sum()), tv_rot90=0.5 * float(np.abs(h - hr).sum()))
    print(json.dumps(out, indent=1))
    with open(os.path.join(RES_DIR, "flow_overlap_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
