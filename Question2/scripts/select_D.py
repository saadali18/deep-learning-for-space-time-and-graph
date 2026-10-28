"""Choose model D among the candidate designs using the VALIDATION split only.

Each candidate was trained with the full model-D protocol (speed augmentation, seed 0, 30 epochs).
Criterion: mean validation E(2) over 1x, 2x and 3x at the final epoch (the quantity Table 5
compares across speeds). The winner's seed-0 run becomes D_s0.
"""
import json
import os
import shutil

from mm.models import D_CANDIDATES
from mm.style import RES_DIR
from mm.train import CKPT_DIR


def main():
    res = {}
    for name in D_CANDIDATES:
        with open(os.path.join(CKPT_DIR, f"Dcand_{name}_s0.json")) as fh:
            log = json.load(fh)
        last = log["epochs"][-1]
        res[name] = dict(params=log["params"], val_E5_1x=last["val_E5_1x"],
                         **{f"val_E2_{k}x": last[f"val_E2_{k}x"] for k in (1, 2, 3)},
                         train_s_per_epoch=log["train_seconds_per_epoch"])
        res[name]["val_E2_mean"] = sum(res[name][f"val_E2_{k}x"] for k in (1, 2, 3)) / 3
    best = min(res, key=lambda n: res[n]["val_E2_mean"])
    for n, r in res.items():
        print(n, {k: f"{v:.4g}" for k, v in r.items()}, "<- chosen" if n == best else "")
    with open(os.path.join(RES_DIR, "model_D_selection.json"), "w") as fh:
        json.dump(dict(candidates=res, chosen=best), fh, indent=1)
    for ext in ("pt", "json"):
        shutil.copy(os.path.join(CKPT_DIR, f"Dcand_{best}_s0.{ext}"), os.path.join(CKPT_DIR, f"D_s0.{ext}"))
    print(best)


if __name__ == "__main__":
    main()
