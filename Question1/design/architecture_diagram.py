"""Q1 architecture diagram for the river forecasting proposal (drawn with matplotlib)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1_architecture.png")

INK, INK2 = "#0b0b0b", "#52514e"
C_SRC, C_ALN, C_PRJ, C_T, C_S, C_HEAD = "#dbe8f8", "#fde6d6", "#efe9fb", "#d6f0e5", "#d9ecf7", "#fbefc9"
E_SRC, E_ALN, E_PRJ, E_T, E_S, E_HEAD = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#2a78d6", "#c98500"


def box(ax, x, y, w, h, title, body, fc, ec, tsize=8.6, bsize=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + 0.008, y + h - 0.012, title, ha="left", va="top", fontsize=tsize, weight="bold", color=INK)
    ax.text(x + 0.008, y + h - 0.040, body, ha="left", va="top", fontsize=bsize, color=INK2, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p, q, label=None, color=INK2, lpos=0.5, dy=0.012, rad=0.0, lsize=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = p[0] + (q[0] - p[0]) * lpos, p[1] + (q[1] - p[1]) * lpos
        ax.text(mx, my + dy, label, ha="center", va="bottom", fontsize=lsize, color=INK, family="DejaVu Sans Mono",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def right(b, f=0.5):
    return (b[0] + b[2], b[1] + b[3] * f)


def left(b, f=0.5):
    return (b[0], b[1] + b[3] * f)


def main():
    fig = plt.figure(figsize=(16, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.985, "River forecasting system: graph-structured space–time forecaster (issued every 10 min, forecasts at t0 + 60 min)",
            fontsize=12, weight="bold", va="top", color=INK)
    ax.text(0.01, 0.958, "B = batch of issuance times;  N = 7 river-graph nodes {Res, S1, T1, S2, U, S3, S4};  H = 72 history steps (12 h at Δ = 10 min);  "
            "F = 6 future steps (t0, t0+60 min];  d = 32 hidden channels",
            fontsize=8.4, va="top", color=INK2)

    # column headers
    for x, t in [(0.01, "1  Data sources"), (0.205, "2  Alignment to the 10-min grid"), (0.425, "3  Projection to node tokens"),
                 (0.618, "4  Temporal + spatial blocks"), (0.835, "5  Forecast head")]:
        ax.text(x, 0.925, t, fontsize=9.5, weight="bold", color=INK)

    # 1 sources
    s_lev = box(ax, 0.01, 0.745, 0.175, 0.16, "Water levels S1–S4, T1",
                "every 10 min (may be late,\nmissing, or noisy)\nhistory only: t ≤ t0", C_SRC, E_SRC)
    s_rain = box(ax, 0.01, 0.555, 0.175, 0.16, "Rainfall R1, R2",
                 "irregular timestamps,\ngauge coordinates known\nhistory only: t ≤ t0", C_SRC, E_SRC)
    s_rel = box(ax, 0.01, 0.335, 0.175, 0.19, "Reservoir releases",
                "recorded releases (past)\n+ announced schedules with\nannouncement time a and\neffective time e;\nuse only a ≤ t0", C_SRC, E_SRC)
    s_static = box(ax, 0.01, 0.17, 0.175, 0.14, "Static river description",
                   "connectivity, along-river\ndistances d_ij, station and\ngauge locations", C_SRC, E_SRC)

    # 2 alignment
    a_lev = box(ax, 0.205, 0.745, 0.2, 0.16, "Snap + mask",
                "snap readings to grid τ = t0−71Δ…t0\n(±5 min); last value carried forward\nwith mask m and age a (min)\n→ X_lev  B×5×72×4",
                C_ALN, E_ALN)
    a_rain = box(ax, 0.205, 0.555, 0.2, 0.16, "Accumulate into bins",
                 "rain depth summed in (τ−Δ, τ];\nmask + age per gauge\n→ X_rain  B×2×72×3", C_ALN, E_ALN)
    a_rel = box(ax, 0.205, 0.335, 0.2, 0.19, "As-of join + step function",
                "past: recorded release (schedule\nonly where no record yet)\nfuture: latest schedule announced\nby t0, held as a step function\n→ X_rel  B×72×4,  X_fut  B×6×3",
                C_ALN, E_ALN)
    a_static = box(ax, 0.205, 0.17, 0.2, 0.14, "Graph construction",
                   "directed adjacency A ∈ {0,1}^{7×7}\nedge lag window from d_ij\nrain-to-node map C ∈ ℝ^{7×2}\n(from sub-catchments)", C_ALN, E_ALN)

    # 3 projection
    p_tok = box(ax, 0.425, 0.48, 0.175, 0.425, "Per-source projection",
                "W_lev: 4→d  on S1–S4, T1\n"
                "W_rain: 3→d, mixed onto\n  nodes by C (B×2×72×d →\n  B×7×72×d)\n"
                "W_rel: 4→d  on Res\n"
                "U (unmonitored) gets no\n  level token, only rain + e_U\n"
                "+ node embedding e_v ∈ ℝ^d\n+ time-of-day / season code\n\n"
                "sum per node and step\n→ Z⁰  B×7×72×d",
                C_PRJ, E_PRJ)
    p_fut = box(ax, 0.425, 0.335, 0.175, 0.12, "Future-release encoder",
                "MLP over X_fut (flattened 6×3)\n→ f  B×d, added to the\nRes node state at t0", C_PRJ, E_PRJ)

    # 4 temporal + spatial
    t_blk = box(ax, 0.618, 0.69, 0.2, 0.215, "Temporal block (T)",
                "reshape B×7×72×d → (B·7)×d×72\ncausal dilated TCN, shared weights,\n6 layers, k = 3, dilation 1…32,\ngated residual units\n(receptive field 127 steps)\n→ H  B×7×72×d",
                C_T, E_T)
    s_blk = box(ax, 0.618, 0.335, 0.2, 0.33, "Spatial block (S), ×4",
                "directed lag-attention message\npassing along river edges j → i:\n"
                "query = node i at step τ\nkeys/values = upstream node j at\n  τ−ℓ, ℓ ∈ [0, L_ji] + travel-\n  time prior (state-dependent lag)\n"
                "h_i,τ ← h_i,τ + MLP([h_i,τ ‖ Σ_j m_ji,τ])\n4 rounds = 4 hops: Res→S1→S2→\n  S3→S4 is within reach\n→ h⁽⁴⁾  B×7×72×d; read at t0",
                C_S, E_S)

    # 5 head
    h_gather = box(ax, 0.835, 0.6, 0.155, 0.305, "Gather + decode",
                   "select S1–S4 rows → B×4×2d\n[h⁽⁴⁾_i(t0) ‖ h_i(t0)]\nshared MLP + station emb.\n→ Δ quantiles  B×4×3\n(monotone: q10 ≤ q50 ≤ q90)\n\n"
                   "Ŷ = y_last + Δ, where y_last\n(B×4) is the latest valid\nlevel taken from X_lev\n(residual on persistence)",
                   C_HEAD, E_HEAD)
    h_out = box(ax, 0.835, 0.335, 0.155, 0.235, "Forecasts at t0 + 60 min",
                "Ŷ  B×4×3\n(station × quantile)\n\nS1, S2, S3, S4 levels:\nmedian + 80 % interval,\nconformal-calibrated;\n+ data-quality flags",
                C_HEAD, E_HEAD)

    # arrows: sources -> alignment
    for s, a in [(s_lev, a_lev), (s_rain, a_rain), (s_rel, a_rel), (s_static, a_static)]:
        arrow(ax, right(s), left(a))
    # alignment -> projection
    arrow(ax, right(a_lev), left(p_tok, 0.88), "B×5×72×4", lpos=0.5)
    arrow(ax, right(a_rain), left(p_tok, 0.5), "B×2×72×3", lpos=0.5)
    arrow(ax, right(a_rel, 0.7), left(p_tok, 0.1), "B×72×4", lpos=0.45)
    arrow(ax, right(a_rel, 0.25), left(p_fut, 0.5), "B×6×3", lpos=0.5, dy=-0.03)
    # static -> river graph inset -> spatial block (A, L_ij); C is used inside the projection block
    arrow(ax, right(a_static, 0.5), (0.432, 0.24), "A, C, L_ij", lpos=0.5)
    arrow(ax, (0.70, 0.296), (0.70, 0.336), None, color="#2a78d6")
    ax.text(0.704, 0.305, "A, L_ij", fontsize=7, family="DejaVu Sans Mono", color=INK)
    # projection -> T
    arrow(ax, right(p_tok, 0.8), left(t_blk, 0.5), "B×7×72×d", lpos=0.45)
    # T -> S
    arrow(ax, (0.718, 0.69), (0.718, 0.665), None)
    ax.text(0.722, 0.672, "B×7×72×d", fontsize=7, family="DejaVu Sans Mono", color=INK)
    # future -> S
    arrow(ax, right(p_fut), left(s_blk, 0.2), "B×d → Res", lpos=0.5, dy=0.012)
    # S -> head
    arrow(ax, right(s_blk, 0.8), left(h_gather, 0.3), "h⁽⁴⁾(t0)", lpos=0.5)
    arrow(ax, right(t_blk, 0.3), left(h_gather, 0.8), "h(t0)", lpos=0.5)
    arrow(ax, (0.9125, 0.6), (0.9125, 0.57), None)
    # inset: river graph
    gx, gy = 0.425, 0.03
    ax.add_patch(FancyBboxPatch((gx, gy), 0.565, 0.265, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc="#f7f7f5", ec="#b5b4ad", lw=1))
    ax.text(gx + 0.01, gy + 0.25, "River graph used by the spatial block (edges point downstream; travel times unknown and state-dependent)",
            fontsize=8.6, weight="bold", va="top", color=INK)
    pos = {"Res": (0.47, 0.17), "S1": (0.56, 0.17), "T1": (0.6, 0.09), "S2": (0.67, 0.17), "U": (0.73, 0.09),
           "S3": (0.79, 0.17), "S4": (0.89, 0.17), "R1": (0.53, 0.235), "R2": (0.76, 0.235)}
    styles = {"Res": ("#d9d9d9", "#52514e"), "U": ("white", "#8a8984"), "R1": ("#dbe8f8", "#2a78d6"),
              "R2": ("#dbe8f8", "#2a78d6"), "T1": ("#d6f0e5", "#1baf7a")}
    for n, (x, y) in pos.items():
        fc, ec = styles.get(n, ("#fbefc9", "#c98500"))
        ls = "--" if n == "U" else "-"
        ax.add_patch(FancyBboxPatch((x - 0.022, y - 0.02), 0.044, 0.04, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc=fc, ec=ec, lw=1.2, linestyle=ls))
        ax.text(x, y, n, ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
    for a, b in [("Res", "S1"), ("S1", "S2"), ("T1", "S2"), ("S2", "S3"), ("U", "S3"), ("S3", "S4")]:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        arrow(ax, (x0 + 0.022 if y0 == y1 else x0 + 0.01, y0 if y0 == y1 else y0 + 0.02),
              (x1 - 0.022 if y0 == y1 else x1 - 0.012, y1 if y0 == y1 else y1 - 0.02), color="#2a78d6")
    for r, targets in [("R1", ["S1", "T1"]), ("R2", ["U", "S3"])]:
        for t in targets:
            (x0, y0), (x1, y1) = pos[r], pos[t]
            ax.add_patch(FancyArrowPatch((x0, y0 - 0.02), (x1, y1 + 0.02), arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                         color="#8a8984", linestyle=(0, (2, 2))))
    ax.text(0.865, 0.085, "■ forecast targets S1–S4\n■ monitored tributary T1\n□ latent node U (no gauge)\n■ rain gauges → nodes\n   via C (dashed)\n■ Res: release input",
            fontsize=7, color=INK2, va="center")
    ax.text(0.435, 0.04, "Reach: 4 spatial rounds = 4 hops (Res → S4).  Lag windows L_ij ≈ d_ij / v_min cap how far back each upstream history is read.",
            fontsize=7.4, color=INK2)

    # loss box
    box(ax, 0.01, 0.03, 0.395, 0.115, "Training objective and calibration",
        "pinball loss over q ∈ {0.1, 0.5, 0.9}, masked where the target is missing, flood-weighted;\n"
        "random masking / delay augmentation of recent inputs; chronological train/val/test split;\n"
        "split-conformal widening of the interval per station on the validation period", "#f2f2f0", "#8a8984")

    fig.savefig(OUT, dpi=220)
    print(OUT)


if __name__ == "__main__":
    main()
