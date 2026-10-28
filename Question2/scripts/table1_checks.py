"""Table 1: checks of the transformation and window functions used by every experiment."""
import json
import os

import torch

from mm.data import mirror, rotate, window

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def bright(v):
    f, _, r, c = (v == 1).nonzero()[0].tolist()
    return (f, r, c)


def frames_of(t):
    return [int(v.flatten()[0]) for v in t]


def fmt(frames):
    if frames == list(range(frames[0], frames[-1] + 1)) and len(frames) > 2:
        return f"{frames[0]}–{frames[-1]}"
    return ", ".join(map(str, frames))


def main():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    rows = []
    lhs, rhs = mirror(mirror(x)), x
    rows.append(("(a) max|M(M(x)) − x|", "0", f"{(lhs - rhs).abs().max().item():g} (torch.equal: {torch.equal(lhs, rhs)})"))
    lhs = rotate(rotate(rotate(rotate(x))))
    rows.append(("(b) max|R^4(x) − x|", "0", f"{(lhs - rhs).abs().max().item():g} (torch.equal: {torch.equal(lhs, rhs)})"))
    lhs, rhs = mirror(rotate(mirror(x))), rotate(x, -1)
    rows.append(("(c) max|MRM(x) − R^−1(x)|", "0", f"{(lhs - rhs).abs().max().item():g} (torch.equal: {torch.equal(lhs, rhs)})"))
    rows.append(("(d) bright pixel of M(p) (frame, row, column)", "(7, 10, 43)", str(bright(mirror(p)))))
    rows.append(("(e) bright pixel of R(p) (frame, row, column)", "(7, 43, 10)", str(bright(rotate(p)))))

    v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
    expected = {1: "1–5 / 6–10", 2: "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", 3: "1, 4, 7, 10, 13 / 16, 19"}
    for label, k in zip("fgh", (1, 2, 3)):
        X, Y = window(v, 1, k)
        rows.append((f"({label}) {k}× input / target frames", expected[k], f"{fmt(frames_of(X))} / {fmt(frames_of(Y))}"))

    for r in rows:
        print(f"{r[0]:<48} {r[1]:<38} {r[2]}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "table1.json"), "w") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
