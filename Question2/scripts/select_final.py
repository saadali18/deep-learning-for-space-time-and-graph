"""Choose the final model between symmetrised D and symmetrised D+B using the VALIDATION split only.

Rule (fixed before D+B finished training): the lower mean validation E(2) over 1x, 2x and 3x,
averaged over seeds 0 and 1, of the symmetrised forecasters. Ties (difference below the seed-to-seed
spread of either model) keep the default, symmetrised D.
"""
import json
import os

from mm.data import load_videos, split
from mm.evaluate import get_device, predict, scores
from mm.registry import SEEDS, load_model
from mm.style import RES_DIR


def val_scores(name, videos, dev):
    out = {}
    for s in SEEDS:
        m = load_model(name, s, dev)
        out[s] = {k: scores(*predict(m, videos, 1, k, dev))["E2"] for k in (1, 2, 3)}
    per_seed = [sum(out[s].values()) / 3 for s in SEEDS]
    return dict(per_seed_E2=out, mean_E2=sum(per_seed) / len(per_seed), spread=abs(per_seed[0] - per_seed[1]))


def main():
    dev = get_device()
    val = split(load_videos(), "val")
    res = {n: val_scores(n, val, dev) for n in ("sym:D", "sym:DB")}
    d, db = res["sym:D"], res["sym:DB"]
    margin = max(d["spread"], db["spread"])
    final = "DB" if d["mean_E2"] - db["mean_E2"] > margin else "D"
    res.update(final=final, margin=margin)
    print(json.dumps(res, indent=1))
    with open(os.path.join(RES_DIR, "final_model_selection.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(final)


if __name__ == "__main__":
    main()
