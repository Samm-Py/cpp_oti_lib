#!/usr/bin/env python3
"""Reduction tree for OTI jets: MPI_Reduce vs MPI_Allreduce.

The classic MPI picture, but each rank holds a *jet* -- a small array of
coefficients [value | d/da | d/db] -- instead of a scalar. The custom
MPI_OTI_SUM op folds the jets coefficient-wise, so the derivatives reduce
*alongside* the value in the same collective:

  TOP    MPI_Reduce    -> the summed jet lands on root (rank 0) only.
  BOTTOM MPI_Allreduce -> the same summed jet lands on every rank.

The point: whichever collective you choose, the op and the result are identical;
Reduce just delivers it to one rank. Colors match oti_jet_slices.png
(value = yellow, d/da = blue, d/db = red).

Usage: plot_reduce_tree.py [out.png]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

VALUE_COLOR = "#F3C969"   # value coefficient  (matches DOMAIN_COLOR)
DA_COLOR = "#73A9D8"      # d/da               (matches e_0)
DB_COLOR = "#E08E79"      # d/db               (matches e_1)
CELL_COLORS = [VALUE_COLOR, DA_COLOR, DB_COLOR]

# per-rank jets [value, d/da, d/db]; each column is a permutation of 1..4 so
# every coefficient sums to a clean 10 -- easy to verify in your head
JETS = [[1, 2, 3], [2, 3, 1], [3, 4, 2], [4, 1, 4]]
TOTAL = [sum(col) for col in zip(*JETS)]   # [10, 10, 10]

CW, CH = 0.62, 0.62      # coefficient cell size
RAD = 0.34               # rank circle radius
GAP = 0.06               # gap between circle and first cell
UNIT = RAD * 2 + GAP + 3 * CW + 0.55   # horizontal span of one rank+jet unit


def draw_jet(ax, x, y, rank, cells, faded=False):
    """Rank circle + 3-cell jet, left-anchored at x (vertical centre y).
    Returns (top_anchor, bottom_anchor) on the unit's centre for edges."""
    a = 0.28 if faded else 1.0
    cx = x + RAD
    ax.add_patch(Circle((cx, y), RAD, facecolor="white", edgecolor="0.35",
                        lw=1.6, zorder=3, alpha=a))
    ax.text(cx, y, str(rank), ha="center", va="center", fontsize=12,
            fontweight="bold", color="0.2", zorder=4, alpha=a)
    bx = cx + RAD + GAP
    for i, val in enumerate(cells):
        x0 = bx + i * CW
        ax.add_patch(Rectangle((x0, y - CH / 2), CW, CH, facecolor=CELL_COLORS[i],
                              edgecolor="black", lw=1.4, zorder=3, alpha=a))
        ax.text(x0 + CW / 2, y, str(val), ha="center", va="center", fontsize=11,
                fontweight="bold", color="0.1", zorder=4, alpha=a)
    centre = bx + 1.5 * CW
    return (centre, y + CH / 2 + 0.04), (centre, y - CH / 2 - 0.04)


def draw_op(ax, x, y, w=3.0, h=0.62):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.02,rounding_size=0.31",
                               facecolor="white", edgecolor="#cf2b2b", lw=2.2,
                               zorder=5))
    ax.text(x, y, "MPI_OTI_SUM", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color="#cf2b2b", zorder=6)
    return (x, y + h / 2), (x, y - h / 2)


def panel(ax, title, all_ranks):
    ax.set_aspect("equal")
    ax.axis("off")
    n = len(JETS)
    x0 = 0.4
    rank_y = 2.55
    op_y = 1.25
    res_y = 0.0

    bots = []
    for r in range(n):
        x = x0 + r * UNIT
        _, bot = draw_jet(ax, x, rank_y, r, JETS[r])
        bots.append(bot)

    span = (n - 1) * UNIT
    op_x = x0 + RAD + 0.06 + 1.5 * CW + span / 2
    op_top, op_bot = draw_op(ax, op_x, op_y)

    # edges: every rank -> op
    for b in bots:
        ax.plot([b[0], op_top[0]], [b[1], op_top[1]], color="0.45", lw=1.2,
                zorder=1)

    # results below
    if all_ranks:
        for r in range(n):
            x = x0 + r * UNIT
            _, _ = draw_jet(ax, x, res_y, r, TOTAL)
            cx = x + RAD + 0.06 + 1.5 * CW
            ax.plot([op_bot[0], cx], [op_bot[1], res_y + CH / 2 + 0.04],
                    color="0.45", lw=1.2, zorder=1)
        # fade the non-root ranks of the *input* row? no -- all contribute.
    else:
        # Reduce: result on root only; show the others greyed to make "root only"
        # explicit.
        for r in range(n):
            x = x0 + r * UNIT
            if r == 0:
                _, _ = draw_jet(ax, x, res_y, r, TOTAL)
                cx = x + RAD + 0.06 + 1.5 * CW
                ax.plot([op_bot[0], cx], [op_bot[1], res_y + CH / 2 + 0.04],
                        color="0.45", lw=1.2, zorder=1)
            else:
                draw_jet(ax, x, res_y, r, ["", "", ""], faded=True)

    ax.set_xlim(-0.3, x0 + span + RAD * 2 + 3 * CW + 0.8)
    ax.set_ylim(res_y - CH / 2 - 0.5, rank_y + CH / 2 + 0.55)
    ax.text(-0.1, rank_y + CH / 2 + 0.5, title, fontsize=13.5,
            fontweight="bold", ha="left", va="bottom")


def main(out_path="mpi_reduce_tree.png"):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.0, 6.4))

    panel(ax_top, "MPI_Reduce  (result on root only)", all_ranks=False)
    panel(ax_bot, "MPI_Allreduce  (result on every rank)", all_ranks=True)

    # shared legend tying cells to coefficients
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=1.2)
               for c in CELL_COLORS]
    fig.legend(handles, ["value $f$", r"$\partial f/\partial a$",
                         r"$\partial f/\partial b$"],
               loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), title="each rank holds a jet "
               "(value + derivatives); MPI_OTI_SUM folds each coefficient")

    fig.suptitle("Reducing OTI jets: the custom op folds value and derivatives "
                 "together", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
