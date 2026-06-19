#!/usr/bin/env python3
"""Schematic for the Halo Exchange (Jacobi) tutorial.

Shows what the program actually does: the steady-state heat field (Laplace,
solved here with the same Jacobi iteration) is decomposed over a 2D Cartesian
grid of ranks, and every iteration each rank swaps a one-cell ghost layer with
its four neighbours. The two halo datatypes are labelled where they act:

  * row halos (between x-neighbours, across the horizontal split): a full row is
    CONTIGUOUS in memory -> count = ny of MPI_OTINUM.
  * column halos (between y-neighbours, across the vertical split): a column is
    STRIDED -> MPI_Type_vector over MPI_OTINUM.

Axes: x (the cart dim-0 / row index) is vertical, y (dim-1 / column index) is
horizontal, matching the contiguous-row / strided-column memory layout.

Usage: plot_diagram.py [out.png]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def solve_heat(n=140, iters=9000):
    """Laplace on the unit square, West (x=0) and South (y=0) walls hot."""
    u = np.zeros((n, n))
    u[0, :] = 1.0    # x = 0  -> West wall (bottom, hot)
    u[:, 0] = 1.0    # y = 0  -> South wall (left, hot)
    for _ in range(iters):
        u[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] +
                                u[1:-1, :-2] + u[1:-1, 2:])
    return u


def label(ax, x, y, text, color="black", fs=10.5, weight="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, fontweight=weight,
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6",
                      alpha=0.92))


def main(out_path="mpi_halo.png"):
    u = solve_heat()

    fig, ax = plt.subplots(figsize=(8.4, 8.6))
    # vertical axis = x (rows / cart dim 0), horizontal = y (cols / cart dim 1)
    ax.imshow(u, origin="lower", extent=(0, 1, 0, 1), cmap="inferno",
              aspect="equal")
    ax.set_xlabel("y   (column index, cart dim 1)", fontsize=11)
    ax.set_ylabel("x   (row index, cart dim 0)", fontsize=11)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    # ---- 2x2 decomposition lines -------------------------------------------
    ax.axhline(0.5, color="white", lw=3.0)
    ax.axvline(0.5, color="white", lw=3.0)

    # ---- rank tile labels (rank = coord0*2 + coord1) -----------------------
    # coord0 = x-block (0 bottom, 1 top); coord1 = y-block (0 left, 1 right)
    for (yc, xc, rk, co) in [(0.25, 0.25, 0, "(0,0)"), (0.75, 0.25, 1, "(0,1)"),
                             (0.25, 0.75, 2, "(1,0)"), (0.75, 0.75, 3, "(1,1)")]:
        label(ax, yc, xc, f"rank {rk}\n{co}", fs=10, weight="bold")

    # ---- row halos: contiguous, between x-neighbours (cross x=0.5) ----------
    for yc in (0.25, 0.75):
        ax.add_patch(FancyArrowPatch((yc, 0.42), (yc, 0.58), arrowstyle="<|-|>",
                                     mutation_scale=16, lw=2.2, color="#39FF14",
                                     zorder=5))
    label(ax, 0.5, 0.50, "row halo  (N/S)\ncontiguous: count = ny",
          color="#1a7a0a", fs=9.5, weight="bold")

    # ---- column halos: strided, between y-neighbours (cross y=0.5) ----------
    for xc in (0.25, 0.75):
        ax.add_patch(FancyArrowPatch((0.42, xc), (0.58, xc), arrowstyle="<|-|>",
                                     mutation_scale=16, lw=2.2, color="#00E5FF",
                                     zorder=5))
    label(ax, 0.5, 0.115, "column halo  (E/W)\nstrided: MPI_Type_vector",
          color="#0077aa", fs=9.5, weight="bold")

    # ---- hot wall annotations ----------------------------------------------
    ax.annotate("hot: West wall (x=0)", xy=(0.5, 0.0), xytext=(0.5, -0.12),
                ha="center", va="top", fontsize=9.5, color="#b30000",
                arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.4))
    ax.annotate("hot: South wall (y=0)", xy=(0.0, 0.5), xytext=(-0.16, 0.5),
                ha="center", va="center", rotation=90, fontsize=9.5,
                color="#b30000",
                arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.4))

    ax.set_title("Halo exchange: 2D rank decomposition of the heat field\n"
                 "each iteration, every rank swaps a ghost layer with its four "
                 "neighbours", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
