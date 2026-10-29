"""Frame grids, GIFs and an overlap detector for the qualitative figures."""
import os

import numpy as np
from PIL import Image, ImageDraw

from .style import GIF_DIR, INK2, plt


def overlap_profile(video):
    """Per-frame overlap of the two digits for a uint8 video (20, 1, 64, 64).

    The digits translate rigidly and stay inside the canvas, so the number of lit pixels
    (> 0.2 intensity) is constant except when the digits overlap and share pixels.
    Returns 1 - lit / max(lit) per frame: ~0 when separate, larger when overlapping.
    """
    lit = (video[:, 0].float() / 255 > 0.2).flatten(1).sum(1).float()
    return (1 - lit / lit.max()).numpy()


def _img(ax, frame, label=None, title=None):
    ax.imshow(np.asarray(frame).squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#cfcfcf")
        s.set_linewidth(0.5)
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, color=INK2, labelpad=4)
    if title:
        ax.set_title(title, fontsize=7.5, color=INK2, pad=2)


def grid(rows, row_labels, col_titles=None, size=0.75, sep_after=None, suptitle=None):
    """rows: list of lists of (64, 64) frames (None = empty cell)."""
    nr, nc = len(rows), max(len(r) for r in rows)
    lab_w, top_h = 1.5, (0.55 if suptitle else 0.3)
    W, H = nc * size * 1.04 + lab_w, nr * size * 1.06 + top_h
    fig, axes = plt.subplots(nr, nc, figsize=(W, H), squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=0.06, left=lab_w / W, right=0.995,
                                              bottom=0.01, top=1 - top_h / H))
    for i, r in enumerate(rows):
        for j in range(nc):
            ax = axes[i, j]
            if j < len(r) and r[j] is not None:
                _img(ax, r[j], row_labels[i] if j == 0 else None,
                     col_titles[j] if (col_titles and i == 0) else None)
            else:
                ax.axis("off")
    if sep_after is not None:  # vertical line between input and future columns
        x = (axes[0, sep_after - 1].get_position().x1 + axes[0, sep_after].get_position().x0) / 2
        y0, y1 = axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1
        fig.add_artist(plt.Line2D([x, x], [y0, y1], color="#eb6834", lw=1.2, transform=fig.transFigure))
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.0)
    return fig


def save_gif(rows, row_labels, path, n_in=5, scale=3):
    """Animated GIF stepping through the five input frames and then the five forecast horizons.

    rows: list of 10-frame sequences (inputs 1-5, future horizons 1-5), shown as stacked rows.
    """
    pad, lab_w, head = 4, 150, 20
    S = 64 * scale
    W = lab_w + S + 2 * pad
    H = head + len(rows) * (S + pad) + pad
    frames = []
    for t in range(10):
        im = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(im)
        d.text((pad, 4), f"input frame {t + 1}" if t < n_in else f"forecast horizon h = {t - n_in + 1}", fill=0)
        for i, seq in enumerate(rows):
            y = head + i * (S + pad)
            d.text((pad, y + S // 2 - 6), row_labels[i], fill=0)
            arr = (np.clip(np.asarray(seq[t]).squeeze(), 0, 1) * 255).astype(np.uint8)
            im.paste(Image.fromarray(arr).resize((S, S), Image.NEAREST), (lab_w, y))
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[400] * 5 + [700] * 5, loop=0)
    return os.path.relpath(path, os.path.dirname(GIF_DIR))
