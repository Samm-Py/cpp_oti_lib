#!/usr/bin/env python3
"""Schematic for the unstructured ghost-exchange (MPI_Type_indexed) rung.

Two ranks, each a shelf of owned nodes plus a small ghost region. EVERY node is a
full OTI_M^N jet, drawn as an oblique tower (value, first-order directions, ... up
to order N) exactly as in the other tutorials -- so one node is one jet and the
whole tower travels as a single element.

The exchange is bidirectional and irregular:
  * rank 0 sends owned nodes 1 and 3 -> rank 2's contiguous ghost block.
  * rank 2 sends owned nodes 10 and 15 -> rank 0's contiguous ghost block.

Usage: plot_diagram.py [out.png]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

R0 = "#3B6CA8"        # rank 0 colour
R2 = "#4E9A57"        # rank 2 colour
SEND_02 = "#C0504D"   # rank 0 -> rank 2 arrows / highlight
SEND_20 = "#2E8B83"   # rank 2 -> rank 0 arrows / highlight
FIGURE_DPI = 200

# OTI_M^N coefficient colours (match oti_jet_slices / mpi_halo)
VAL, FIRST, SECOND, TOP = "#F3C969", "#73A9D8", "#9BCB8C", "#70C1B3"

# tower geometry (one node)
SX, DX, DY, DZ = 0.52, 0.22, 0.18, 0.24
LEVELS = [(0.0, VAL), (1.0, FIRST), (2.0, FIRST), (3.4, FIRST),
          (4.6, SECOND), (6.0, TOP)]
DOTS_Z = [2.7, 5.3]
TOWER_TOP_Z = LEVELS[-1][0]

GX = 1.08             # column pitch
GAP = 0.55            # extra gap before the ghost region


def colx(col):
    return 1.8 + col * GX + (GAP if col >= 6 else 0.0)


def Cpt(bx, bz, u, v, z):
    return (bx + SX * u + DX * v, bz + z * DZ + DY * v)


def node_tower(ax, col, bz, label, base_edge, base_lw, faded):
    """Draw one OTI_M^N jet as a stacked oblique tower."""
    bx = colx(col)
    for z, color in LEVELS:
        pts = [Cpt(bx, bz, 0, 0, z), Cpt(bx, bz, 1, 0, z),
               Cpt(bx, bz, 1, 1, z), Cpt(bx, bz, 0, 1, z)]
        if z == 0.0:
            ec, lw = base_edge, base_lw
        else:
            ec, lw = "0.3", 0.8
        a, ls = (0.45, (0, (2, 2))) if faded else (0.95, "solid")
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=ec,
                             lw=lw, alpha=a, linestyle=ls, zorder=3 + int(z)))
    for z in DOTS_Z:
        dxc, dyc = Cpt(bx, bz, 0.5, 0.5, z)
        ax.text(dxc, dyc, r"$\vdots$", fontsize=7.5, ha="center", va="center",
                color="0.45", zorder=20)
    bxc, byc = Cpt(bx, bz, 0.5, 0.5, 0.0)
    ax.text(bxc, byc, label, fontsize=8.5, ha="center", va="center",
            fontweight="bold", color="black", zorder=21)


def base_bottom(col, bz):
    return Cpt(colx(col), bz, 0.5, 0.0, 0.0)


def tower_top(col, bz):
    return Cpt(colx(col), bz, 0.5, 1.0, TOWER_TOP_Z)


def label_specimen(ax, col, bz):
    """Annotate one tower's planes with the OTI_M^N coefficient names."""
    bx = colx(col)
    items = [(0.0, r"$c_{\mathbf{0}}$  (value)"), (1.0, r"$c_{e_0}$"),
             (2.0, r"$c_{e_1}$"), (3.4, r"$c_{e_{M-1}}$"),
             (4.6, r"$\{c_\alpha\},\,|\alpha|{=}2$"),
             (6.0, r"$\{c_\alpha\},\,|\alpha|{=}N$")]
    n = len(items)
    for i, (z, lab) in enumerate(items):
        ly = bz + 0.05 + (2.0 - 0.05) * i / (n - 1)
        px, py = Cpt(bx, bz, 0.0, 0.5, z)
        ax.plot([1.55, px], [ly, py], color="0.6", lw=0.7, zorder=19)
        ax.text(1.45, ly, lab, fontsize=7.8, ha="right", va="center",
                color="0.2", zorder=20)


def main(out_path="mpi_unstructured.png"):
    fig, ax = plt.subplots(figsize=(13.2, 8.6))
    ax.set_xlim(0, 12.7); ax.set_ylim(0, 7.4)
    ax.set_aspect("equal"); ax.axis("off")

    TY, BY = 4.45, 1.15           # baseline (value plane) of the two shelves

    # ---- rank 0 shelf (top): owned 0..5, ghosts 10,15 -------------------
    for i, g in enumerate(range(0, 6)):
        hl = g in (1, 3)
        node_tower(ax, i, TY, str(g), SEND_02 if hl else R0,
                   2.6 if hl else 1.5, faded=False)
    for k, g in enumerate((10, 15)):
        node_tower(ax, 6 + k, TY, str(g), SEND_20, 2.4, faded=True)
    ax.text(colx(2) + SX / 2, TY + 2.05, "rank 0 — owns 0–5", fontsize=10,
            ha="center", va="bottom", color=R0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.75))
    ax.text(colx(7) + 1.0, TY + 0.85, "ghosts\n(from rank 2)", fontsize=8.5,
            ha="left", va="center", color=R2)

    # ---- rank 2 shelf (bottom): owned 10..15, ghosts 1,3 ----------------
    for i, g in enumerate(range(10, 16)):
        hl = g in (10, 15)
        node_tower(ax, i, BY, str(g), SEND_20 if hl else R2,
                   2.6 if hl else 1.5, faded=False)
    for k, g in enumerate((1, 3)):
        node_tower(ax, 6 + k, BY, str(g), SEND_02, 2.4, faded=True)
    ax.text(colx(2) + SX / 2, BY + 2.05, "rank 2 — owns 10–15", fontsize=10,
            ha="center", va="bottom", color=R2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.75))
    ax.text(colx(7) + 1.0, BY + 0.85, "ghosts\n(from rank 0)", fontsize=8.5,
            ha="left", va="center", color=R0)
    label_specimen(ax, 0, TY)

    # ---- exchange arrows (in the gap between the shelves) ---------------
    def arrow(c_from, c_to, color, rad, down):
        if down:                        # rank 0 (top) -> rank 2 ghost (bottom)
            p0, p1 = base_bottom(c_from, TY), tower_top(c_to, BY)
        else:                           # rank 2 (bottom) -> rank 0 ghost (top)
            p0, p1 = tower_top(c_from, BY), base_bottom(c_to, TY)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                     lw=2.1, color=color, zorder=18,
                                     connectionstyle=f"arc3,rad={rad}"))

    arrow(1, 6, SEND_02, -0.15, down=True)    # r0 node 1 -> r2 ghost 1
    arrow(3, 7, SEND_02, -0.08, down=True)    # r0 node 3 -> r2 ghost 3
    arrow(0, 6, SEND_20, 0.15, down=False)    # r2 node 10 -> r0 ghost 10
    arrow(5, 7, SEND_20, 0.08, down=False)    # r2 node 15 -> r0 ghost 15

    # ---- arrow legend ---------------------------------------------------
    ax.add_patch(FancyArrowPatch((1.7, 0.7), (2.5, 0.7), arrowstyle="-|>",
                                 mutation_scale=14, lw=2.1, color=SEND_02))
    ax.text(2.65, 0.7, "rank 0 $\\to$ rank 2  (owned nodes 1, 3)",
            fontsize=9, ha="left", va="center", color="0.2")
    ax.add_patch(FancyArrowPatch((6.6, 0.7), (7.4, 0.7), arrowstyle="-|>",
                                 mutation_scale=14, lw=2.1, color=SEND_20))
    ax.text(7.55, 0.7, "rank 2 $\\to$ rank 0  (owned nodes 10, 15)",
            fontsize=9, ha="left", va="center", color="0.2")

    fig.suptitle("Unstructured exchange: each node is an "
                 r"$\mathrm{OTI}_M^N$ jet, and MPI_Type_indexed ships the "
                 "scattered ones whole", fontsize=12.5, fontweight="bold",
                 y=0.96)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
