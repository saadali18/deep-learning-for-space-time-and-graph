"""Table 2 and Figure 2: model A versus Blank and Copy-last on the original test clips (1x, t = 1)."""
import json
import os

from mm.evaluate import get_device
from mm.registry import SEEDS, load_model, metrics_for, test_videos, train_log
from mm.style import RES_DIR
from scripts.figs import fig2, pick_fig2_videos, store_selection


def main():
    rows = {}
    for s in SEEDS:
        rows[f"A_s{s}"] = dict(metrics_for("A", s, kind="horizon"), params=train_log("A", s)["params"])
    for n in ("Blank", "Copy-last"):
        rows[n] = dict(metrics_for(n, 0, kind="horizon"), params=0)
    print(json.dumps(rows, indent=1))
    with open(os.path.join(RES_DIR, "table2.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    vids = test_videos()
    idxs = pick_fig2_videos(vids)
    store_selection("fig2", idxs)
    print("figure 2 videos", [9000 + i for i in idxs])
    fig2(load_model("A", 0, get_device()), vids, idxs, "fig2_modelA_examples.png")


if __name__ == "__main__":
    main()
