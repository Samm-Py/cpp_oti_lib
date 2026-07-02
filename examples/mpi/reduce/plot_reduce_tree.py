#!/usr/bin/env python3
"""Tower-style diagram for MPI reductions over OTI values.

The picture intentionally stays agnostic to the reduction operator. It shows the
MPI buffer contract: matching OTI elements on each rank are combined into matching
output elements. MPI_Reduce delivers those outputs only to root; MPI_Allreduce
delivers the same outputs to every rank.

Usage: plot_reduce_tree.py [out.png]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

DARK = "#2b2b2b"
MUTED = "#696969"
RED = "#c94f4f"
TEAL = "#2f978b"
YELLOW = "#F3C969"
BLUE = "#73A9D8"
GREEN = "#9BCB8C"
TOP = "#70C1B3"

# Same OTI tower geometry as unstructured/plot_diagram.py.
SX, DX, DY, DZ = 0.52, 0.22, 0.18, 0.24
LEVELS = [(0.0, YELLOW), (1.0, BLUE), (2.0, BLUE), (3.4, BLUE),
          (4.6, GREEN), (6.0, TOP)]
DOTS_Z = [2.7, 5.3]

def tower(ax, x, y, label, scale=1.0, faded=False, base=YELLOW):
    """Draw one OTI_M^N tower using the unstructured-exchange geometry."""
    alpha = 0.28 if faded else 0.95
    line_style = (0, (2, 2)) if faded else "solid"

    def pt(u, v, z):
        return (x + scale * (SX * u + DX * v),
                y + scale * (z * DZ + DY * v))

    for z, color in LEVELS:
        pts = [
            pt(0, 0, z),
            pt(1, 0, z),
            pt(1, 1, z),
            pt(0, 1, z),
        ]
        lw = 1.0 if z == 0.0 else 0.8
        ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                             edgecolor=DARK, lw=lw, alpha=alpha,
                             linestyle=line_style, zorder=3 + int(z)))
    for z in DOTS_Z:
        dx0, dy0 = pt(0.5, 0.5, z)
        ax.text(dx0, dy0, r"$\vdots$", fontsize=7.5 * scale / 0.72,
                ha="center", va="center", color="0.45", alpha=alpha,
                zorder=20)
    bx, by = pt(0.5, 0.5, 0.0)
    ax.text(bx, by - 0.42 * scale, label, ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=DARK, alpha=alpha,
            zorder=21)


def arrow(ax, start, end, color=MUTED, rad=0.0, lw=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}",
                                 zorder=1))


def op_box(ax, x, y, text):
    ax.add_patch(FancyBboxPatch((x - 0.98, y - 0.28), 1.96, 0.56,
                                boxstyle="round,pad=0.025,rounding_size=0.08",
                                facecolor="white", edgecolor=RED, lw=2.0,
                                zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=12,
            fontweight="bold", color=RED, zorder=5)


def rank_row(ax, x, y, rank, labels):
    ax.text(x + 1.48, y + 1.58, rank, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=DARK)
    for i, label in enumerate(labels):
        tower(ax, x + i * 1.25, y, label, scale=0.72)


def out_row(ax, x, y, title, labels, faded=False):
    ax.text(x + 1.48, y + 1.68, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=DARK,
            alpha=0.32 if faded else 1.0)
    for i, label in enumerate(labels):
        tower(ax, x + i * 1.25, y, label, scale=0.72, faded=faded)


def draw_reduce(ax, y):
    top_y = y + 1.0
    bottom_y = y - 1.25

    rank_row(ax, 1.05, top_y, "rank 0",
             [r"$J_0^{(0)}$", r"$J_1^{(0)}$", r"$J_2^{(0)}$"])
    rank_row(ax, 1.05, bottom_y, "rank 1",
             [r"$J_0^{(1)}$", r"$J_1^{(1)}$", r"$J_2^{(1)}$"])
    op_box(ax, 5.95, y + 0.45, "Reduce")
    out_row(ax, 8.15, y + 0.05, "root",
            [r"$G_0$", r"$G_1$", r"$G_2$"])

    arrow(ax, (4.45, y + 1.18), (4.90, y + 0.66), color=RED, rad=-0.08)
    arrow(ax, (4.45, y - 0.28), (4.90, y + 0.24), color=TEAL, rad=0.08)
    arrow(ax, (6.95, y + 0.45), (7.95, y + 0.45), color=MUTED)


def draw_allreduce(ax, y):
    top_y = y + 1.0
    bottom_y = y - 1.25

    rank_row(ax, 1.05, top_y, "rank 0",
             [r"$J_0^{(0)}$", r"$J_1^{(0)}$", r"$J_2^{(0)}$"])
    rank_row(ax, 1.05, bottom_y, "rank 1",
             [r"$J_0^{(1)}$", r"$J_1^{(1)}$", r"$J_2^{(1)}$"])
    op_box(ax, 5.95, y + 0.45, "Allreduce")
    out_row(ax, 8.15, top_y, "rank 0",
            [r"$G_0$", r"$G_1$", r"$G_2$"])
    out_row(ax, 8.15, bottom_y, "rank 1",
            [r"$G_0$", r"$G_1$", r"$G_2$"])

    arrow(ax, (4.45, y + 1.18), (4.90, y + 0.66), color=RED, rad=-0.08)
    arrow(ax, (4.45, y - 0.28), (4.90, y + 0.24), color=TEAL, rad=0.08)
    arrow(ax, (6.95, y + 0.55), (7.95, top_y + 0.45), color=MUTED, rad=0.05)
    arrow(ax, (6.95, y + 0.30), (7.92, bottom_y + 0.45), color=MUTED, rad=-0.15)


def main(out_path="mpi_reduce_tree.png"):
    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    ax.set_xlim(0, 12.1)
    ax.set_ylim(-5.0, 5.85)
    ax.axis("off")

    ax.text(5.7, 5.18, "Reducing OTI buffers", ha="center", va="center",
            fontsize=17, fontweight="bold", color=DARK)

    draw_reduce(ax, 2.25)
    draw_allreduce(ax, -2.75)

    ax.text(6.05, -4.62,
            r"$G_i=\mathrm{op}(J_i^{(0)},J_i^{(1)},\ldots)$",
            ha="center", va="center", fontsize=12.5, color=DARK)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
