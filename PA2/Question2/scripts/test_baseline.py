"""Testing the baseline: Table 3, Figures 3-4 (mirror/rotation) and Table 4, Figure 5 (speed)."""
import json
import os

import numpy as np

from mm.evaluate import get_device
from mm.registry import SEEDS, load_model, metrics_for, per_video, test_videos
from mm.style import RES_DIR
from scripts.figs import fig3_panel, fig4, fig5, pick_fig5_video, store_selection


def main():
    rows = {f"A_s{s}": metrics_for("A", s) for s in SEEDS}
    rows.update({n: metrics_for(n, 0) for n in ("Blank", "Copy-last")})
    with open(os.path.join(RES_DIR, "table3_4.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    keys = ["E5_1x", "CM", "EM", "CR", "ER", "E2_1x", "E2_2x", "E2_3x", "E5_2x"]
    for n, r in rows.items():
        print(n, {k: f"{r[k]:.3g}" for k in keys})

    vids = test_videos()
    model = load_model("A", 0, get_device())
    pv = per_video("A", 0)
    top = {tn: [int(i) for i in np.argsort(-pv[f"C{tn}"])[:2]] for tn in ("M", "R")}
    store_selection("fig3", top)
    gifs, marks = {}, {}
    for tn in ("M", "R"):
        marks[tn] = []
        for r, idx in enumerate(top[tn]):
            gif, c, e = fig3_panel(model, vids, idx, tn, "fig3_modelA", "Model A (seed 0)")
            gifs[f"{tn}_{9000 + idx}"] = dict(gif=gif, C=c, E=e)
            marks[tn].append((idx, f"video {9000 + idx}"))
    with open(os.path.join(RES_DIR, "fig3_gifs_modelA.json"), "w") as fh:
        json.dump(gifs, fh, indent=1)
    print(json.dumps(gifs, indent=1))
    fig4(pv, marks, "fig4_consistency_hist_modelA.png", "Per-video consistency gaps of model A (seed 0) on the 1,000 test videos")

    idx = pick_fig5_video(vids)
    store_selection("fig5", idx)
    print("figure 5 video", 9000 + idx)
    fig5(model, vids, idx, "fig5_speed_modelA.png", "Model A (seed 0)")


if __name__ == "__main__":
    main()
