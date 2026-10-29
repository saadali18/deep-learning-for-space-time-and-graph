"""Table 5 (models A-D), Table 6 (symmetrised models) and Figure 6 (final model on the earlier videos).

usage: python -m scripts.final_tables            (final model = symmetrised D)
"""
import argparse
import json
import os

from mm.evaluate import get_device
from mm.registry import SEEDS, load_model, metrics_for, test_videos
from mm.style import RES_DIR
from scripts.figs import fig3_panel, fig5, selections

KEYS = ["E5_1x", "CM", "EM", "CR", "ER", "E2_1x", "E2_2x", "E2_3x", "params", "train_s_per_epoch", "inference_s"]


def table(names):
    out = {}
    for n in names:
        for s in SEEDS:
            out[f"{n}_s{s}"] = metrics_for(n, s)
            print(n, s, {k: f"{out[f'{n}_s{s}'][k]:.3g}" for k in KEYS})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="D", help="underlying model of the final symmetrised forecaster")
    ap.add_argument("--with-db", action="store_true", help="also report DB (D+B) in Table 5")
    a = ap.parse_args()

    t5 = table(["A", "B", "C", "D"] + (["DB"] if a.with_db else []))
    with open(os.path.join(RES_DIR, "table5.json"), "w") as fh:
        json.dump(t5, fh, indent=1)
    sym = ["sym:B", "sym:D"] + ([f"sym:{a.final}"] if a.final not in ("B", "D") else [])
    t6 = table(sym)
    with open(os.path.join(RES_DIR, "table6.json"), "w") as fh:
        json.dump(dict(rows=t6, final=f"sym:{a.final}"), fh, indent=1)

    vids = test_videos()
    final = load_model(f"sym:{a.final}", 0, get_device())
    sel = selections()
    gifs = {}
    for tn in ("M", "R"):
        for idx in sel["fig3"][tn]:
            gif, c, e = fig3_panel(final, vids, idx, tn, "fig6a_final", f"Final model (symmetrised {a.final}, seed 0)")
            gifs[f"{tn}_{9000 + idx}"] = dict(gif=gif, C=c, E=e)
    with open(os.path.join(RES_DIR, "fig6_gifs_final.json"), "w") as fh:
        json.dump(gifs, fh, indent=1)
    print(json.dumps(gifs, indent=1))
    fig5(final, vids, sel["fig5"], "fig6b_speed_final.png", f"Final model (symmetrised {a.final}, seed 0)")

    # the non-symmetrised underlying model on the same panels, for the Q2.16 attribution
    under = load_model(a.final, 0, get_device())
    for tn in ("M", "R"):
        for idx in sel["fig3"][tn]:
            fig3_panel(under, vids, idx, tn, f"extra_{a.final}", f"Model {a.final} without symmetrisation (seed 0)")
    fig5(under, vids, sel["fig5"], f"extra_{a.final}_speed.png", f"Model {a.final} without symmetrisation (seed 0)")


if __name__ == "__main__":
    main()
