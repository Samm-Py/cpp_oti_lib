#!/usr/bin/env python3
"""Schematic for the Independent Evaluation (Gather) tutorial.

Shows what the program actually does: the global domain of
f(x, y) = sin(x)*exp(y) is split into contiguous blocks, one per MPI rank; each
rank evaluates the function on *its* block alone (no communication); then one
MPI_Gatherv assembles every block on rank 0. A callout shows that each grid point
is not a scalar but a jet -- the value plus its derivative coefficients -- which
is the single element the committed MPI datatype moves.

Usage: plot_diagram.py [out.png]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RANKS = 4
RANK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def field(n=240):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")  # row index = x, like the toy
    return np.sin(X) * np.exp(Y)


def main(out_path="mpi_gather.png"):
    f = field()
    n = f.shape[0]
    bounds = [round(r * n / RANKS) for r in range(RANKS + 1)]

    fig = plt.figure(figsize=(12, 6.0))
    extent = [0, 1, 0, 1]
    cmap, vmin, vmax = "viridis", f.min(), f.max()

    # explicit axes positions [left, bottom, width, height] in figure coords
    ax_left = fig.add_axes([0.10, 0.34, 0.32, 0.52])
    ax_right = fig.add_axes([0.56, 0.34, 0.32, 0.52])
    cax = fig.add_axes([0.895, 0.34, 0.015, 0.52])

    # Display f with the row index (i = x) on the vertical axis, so the
    # horizontal strips below are exactly the contiguous row-blocks that the flat
    # block partition assigns to each rank.
    # ---- left: domain decomposed into per-rank blocks ----------------------
    ax_left.imshow(f, origin="lower", extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")
    for r in range(RANKS):
        y0, y1 = bounds[r] / n, bounds[r + 1] / n
        if r < RANKS - 1:
            ax_left.axhline(y1, color="white", lw=2.5)
        ax_left.add_patch(plt.Rectangle((0, y0), 1, y1 - y0, fill=False,
                                        edgecolor=RANK_COLORS[r], lw=3))
        ax_left.text(-0.06, (y0 + y1) / 2, f"rank {r}", ha="right", va="center",
                     fontsize=12, fontweight="bold", color=RANK_COLORS[r])
    ax_left.set_title("1.  Decompose the domain", fontsize=12.5,
                      fontweight="bold", pad=8)
    ax_left.set_xlabel("y")
    ax_left.set_ylabel("x")
    ax_left.set_xticks([0, 0.5, 1])
    ax_left.set_yticks([0, 0.5, 1])

    # ---- right: gathered result on rank 0 ----------------------------------
    im = ax_right.imshow(f, origin="lower", extent=extent, cmap=cmap,
                         vmin=vmin, vmax=vmax, aspect="auto")
    ax_right.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                     edgecolor="black", lw=2.5))
    ax_right.set_title("3.  Gather to rank 0", fontsize=12.5,
                       fontweight="bold", pad=8)
    ax_right.set_xlabel("y")
    ax_right.set_ylabel("x")
    ax_right.set_xticks([0, 0.5, 1])
    ax_right.set_yticks([0, 0.5, 1])
    fig.colorbar(im, cax=cax, label="f(x, y)")

    # ---- arrow between the panels ------------------------------------------
    arrow = FancyArrowPatch((0.435, 0.585), (0.555, 0.585),
                            transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=28,
                            lw=2.6, color="#222222")
    fig.add_artist(arrow)
    fig.text(0.495, 0.64, "MPI_Gatherv", ha="center", va="bottom", fontsize=11,
             fontweight="bold")
    fig.text(0.495, 0.56, "counts in\njets, not bytes", ha="center", va="top",
             fontsize=8.5, style="italic", color="#444444")

    # ---- bottom callout: one grid point is a jet ---------------------------
    fig.text(0.07, 0.155, "2.  Evaluate locally", fontsize=12,
             fontweight="bold", ha="left", va="center")
    fig.text(0.07, 0.082, r"each rank evaluates $f(x,y)=\sin x\;e^{y}$ on its "
             "own block —\nno communication during the compute", fontsize=10,
             ha="left", va="center", style="italic")

    box = FancyBboxPatch((0.50, 0.025), 0.40, 0.165, transform=fig.transFigure,
                         boxstyle="round,pad=0.012", fc="#fdf6e3", ec="#333333",
                         lw=1.4, zorder=5)
    fig.add_artist(box)
    fig.text(0.52, 0.155, "what moves: one grid point is a jet, not a scalar",
             fontsize=10, fontweight="bold", ha="left", va="center", zorder=6)
    fig.text(0.52, 0.085,
             r"$[\;f,\;\; \partial_x f,\;\; \partial_y f,\;\; "
             r"\partial_{xx} f,\;\; \partial_{xy} f,\;\; \partial_{yy} f\;]$"
             "\n= ncoeffs doubles  =  one committed MPI_OTINUM element",
             fontsize=10, ha="left", va="center", zorder=6)

    fig.savefig(out_path, dpi=130)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
