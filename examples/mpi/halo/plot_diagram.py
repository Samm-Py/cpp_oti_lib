#!/usr/bin/env python3
"""Generic OTI_M^N halo-exchange schematic for the Jacobi tutorial.

A 2D process decomposition (2x2) of the structured grid, extended into an oblique
stack using the same visual grammar as toy/plot_jet_slices.py. The value,
individual first-order directions, and representative higher-order groups are
shown as separate planes. Every grid cell carries the complete tower, so a halo
exchange ships one whole OTI_M^N jet per ghost cell.

The top layer is annotated with the four-neighbour exchange:
  * N<->S (between vertically adjacent ranks): a row is CONTIGUOUS in memory
    -> count = ny of MPI_OTINUM.
  * E<->W (between horizontally adjacent ranks): a column is STRIDED
    -> MPI_Type_vector over MPI_OTINUM.

Usage: plot_diagram.py [out.png]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

# oblique (cavalier) projection: local (u, v) at height z -> screen
SX = 3.3
DX, DY = 1.35, 1.0
DZ = 1.05
CELLS = 8
FIGURE_DPI = 220

DOMAIN_COLOR = "#F3C969"
FIRST_ORDER_COLOR = "#73A9D8"
SECOND_ORDER_COLOR = "#9BCB8C"
TOP_ORDER_COLOR = "#70C1B3"
LAYERS = [
    (0.0, DOMAIN_COLOR, r"$|\alpha|=0:\quad c_{\mathbf{0}}=u$"),
    (1.0, FIRST_ORDER_COLOR, r"$c_{e_0}$"),
    (2.0, FIRST_ORDER_COLOR, r"$c_{e_1}$"),
    (4.0, FIRST_ORDER_COLOR, r"$c_{e_{M-1}}$"),
    (5.5, SECOND_ORDER_COLOR,
     r"$|\alpha|=2:\quad \{c_\alpha\}$"),
    (8.0, TOP_ORDER_COLOR,
     r"$|\alpha|=N:\quad \{c_\alpha\}$"),
]
FIRST_ORDER_DOT_LEVEL = 3.0
ORDER_DOT_LEVEL = 6.75

NS_COLOR = "#0b7a0b"   # row halo (contiguous)
EW_COLOR = "#0077aa"   # column halo (strided)


def corner(u, v, z, x0):
    return (x0 + SX * u + DX * v, z + DY * v)


def plane_polygon(z, x0):
    return [corner(0, 0, z, x0), corner(1, 0, z, x0),
            corner(1, 1, z, x0), corner(0, 1, z, x0)]


def draw_ghost_bands(ax, z, x0):
    """The one-cell ghost ring around every tile. This 2x2 is a representative
    slice of a larger decomposition, so every edge -- the internal cross and the
    outer perimeter alike -- is a ghost layer exchanged with a neighbour. All one
    neutral grey, drawn opaque so it does not blend with the tile colours."""
    gc = 1.0 / CELLS
    GHOST = "#CFCFCF"
    poly = lambda pts: Polygon(pts, closed=True, edgecolor="none",
                               facecolor=GHOST, zorder=2.6)
    bands = [
        # internal cross
        [corner(0.5 - gc, 0, z, x0), corner(0.5 + gc, 0, z, x0),
         corner(0.5 + gc, 1, z, x0), corner(0.5 - gc, 1, z, x0)],
        [corner(0, 0.5 - gc, z, x0), corner(1, 0.5 - gc, z, x0),
         corner(1, 0.5 + gc, z, x0), corner(0, 0.5 + gc, z, x0)],
        # outer perimeter
        [corner(0, 0, z, x0), corner(gc, 0, z, x0), corner(gc, 1, z, x0), corner(0, 1, z, x0)],
        [corner(1 - gc, 0, z, x0), corner(1, 0, z, x0), corner(1, 1, z, x0), corner(1 - gc, 1, z, x0)],
        [corner(0, 0, z, x0), corner(1, 0, z, x0), corner(1, gc, z, x0), corner(0, gc, z, x0)],
        [corner(0, 1 - gc, z, x0), corner(1, 1 - gc, z, x0), corner(1, 1, z, x0), corner(0, 1, z, x0)]]
    for b in bands:
        ax.add_patch(poly(b))


def draw_layer(ax, z, x0, color, top=False, labels=False, layer_text=None):
    # The coefficient/order color fills every rank tile; rank ownership is shown
    # by the process cross and labels, matching the gather tutorial's tower.
    for uh, vh in ((0, 0), (0, 1), (1, 0), (1, 1)):
        u0, u1 = uh * 0.5, uh * 0.5 + 0.5
        v0, v1 = vh * 0.5, vh * 0.5 + 0.5
        quad = [corner(u0, v0, z, x0), corner(u1, v0, z, x0),
                corner(u1, v1, z, x0), corner(u0, v1, z, x0)]
        ax.add_patch(Polygon(quad, closed=True, facecolor=color,
                             alpha=0.80 if top else 0.72, edgecolor="none",
                             zorder=2))
    # faint cell grid
    for t in np.linspace(0, 1, CELLS + 1):
        a0, a1 = corner(t, 0, z, x0), corner(t, 1, z, x0)
        b0, b1 = corner(0, t, z, x0), corner(1, t, z, x0)
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="white", lw=0.5,
                alpha=0.5, zorder=2.5)
        ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color="white", lw=0.5,
                alpha=0.5, zorder=2.5)
    # bold process cross (the partition boundaries)
    c0, c1 = corner(0.5, 0, z, x0), corner(0.5, 1, z, x0)
    d0, d1 = corner(0, 0.5, z, x0), corner(1, 0.5, z, x0)
    ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color="black", lw=2.0, zorder=3)
    ax.plot([d0[0], d1[0]], [d0[1], d1[1]], color="black", lw=2.0, zorder=3)
    # ghost-cell ring on every layer (opaque so it never blends with the tiles)
    draw_ghost_bands(ax, z, x0)
    # outline
    ax.add_patch(Polygon(plane_polygon(z, x0), closed=True, fill=False,
                         edgecolor="black", lw=1.8, zorder=3))

    if labels:
        names = {(0, 0): "rank 0", (0, 1): "rank 1", (1, 0): "rank 2",
                 (1, 1): "rank 3"}
        for (uh, vh), nm in names.items():
            cx, cy = corner(uh * 0.5 + 0.25, vh * 0.5 + 0.25, z, x0)
            ax.text(cx, cy, nm, fontsize=10, fontweight="bold", color="black",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                              alpha=0.6))
    if top:
        annotate_top(ax, z, x0)
    if layer_text and layer_text.startswith("$c_"):
        cx, cy = corner(0.5, 0.5, z, x0)
        ax.text(cx, cy, layer_text, fontsize=11, color="#245b85",
                fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.62))


def annotate_top(ax, z, x0):
    gc = 1.0 / CELLS
    def arrow(u0, v0, u1, v1, color, lw=2.1):
        ax.add_patch(FancyArrowPatch(corner(u0, v0, z, x0),
                                     corner(u1, v1, z, x0),
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=lw, color=color, zorder=7))

    # E<->W exchange (column halo, strided). Each shared edge has two distinct
    # sends: the rightmost interior column of the left rank fills the right
    # rank's left ghost column, and vice versa. Offset the arrows in v so both
    # transfers remain visible.
    for vc in (0.25, 0.75):
        arrow(0.5 - 1.35 * gc, vc - 0.045,
              0.5 + 0.55 * gc, vc - 0.045, EW_COLOR, lw=1.9)
        arrow(0.5 + 1.35 * gc, vc + 0.045,
              0.5 - 0.55 * gc, vc + 0.045, EW_COLOR, lw=1.9)

    # N<->S exchange (row halo, contiguous), likewise shown as two sends into
    # the opposite rank's ghost row.
    for uc in (0.25, 0.75):
        arrow(uc - 0.045, 0.5 - 1.35 * gc,
              uc - 0.045, 0.5 + 0.55 * gc, NS_COLOR, lw=1.9)
        arrow(uc + 0.045, 0.5 + 1.35 * gc,
              uc + 0.045, 0.5 - 0.55 * gc, NS_COLOR, lw=1.9)

    # This 2x2 is an interior slice. Simple outward arrows indicate that the
    # same paired exchange continues with ranks beyond the illustrated slice.
    for vc in (0.25, 0.75):
        arrow(0.04, vc, -0.12, vc, EW_COLOR, lw=1.8)
        arrow(0.96, vc, 1.12, vc, EW_COLOR, lw=1.8)
    for uc in (0.25, 0.75):
        arrow(uc, 0.04, uc, -0.12, NS_COLOR, lw=1.8)
        arrow(uc, 0.96, uc, 1.12, NS_COLOR, lw=1.8)

    # "ghost ring" callout, pointing at an outer exchanged ghost edge.
    gx, gy = corner(0.5, 1 - gc, z, x0)
    ax.annotate("ghost ring\n(all edges exchanged)", xy=(gx, gy),
                xytext=(gx - 2.0, gy + 0.95), fontsize=9, color="0.25",
                ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.1))


def main(out_path="mpi_halo.png"):
    fig, ax = plt.subplots(figsize=(14.5, 9.5))
    ax.set_aspect("equal")
    ax.axis("off")
    x0 = 0.0
    top_z = LAYERS[-1][0] * DZ

    # dashed projection lines at the footprint corners + the cross centre
    for (u, v) in [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]:
        xb, yb = corner(u, v, 0.0, x0)
        xt, yt = corner(u, v, top_z, x0)
        ax.plot([xb, xt], [yb, yt], ls=(0, (4, 3)), color="0.6", lw=0.9,
                zorder=1)

    for i, (level, color, label) in enumerate(LAYERS):
        z = level * DZ
        draw_layer(ax, z, x0, color, top=(i == len(LAYERS) - 1),
                   labels=(i == 0), layer_text=label)
        # c_e0 and c_e1 are already written inside their planes. The final
        # displayed first-order plane gets the group-level annotation, matching
        # the gather tutorial.
        if level in (1.0, 2.0):
            continue
        br = corner(1.0, 1.0, z, x0)
        ax.annotate("", xy=(br[0] + 0.85, br[1] + 0.12), xytext=br, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
        if level == 4.0:
            text = (label + "\n" + r"$M$ first-order coefficient fields")
        elif level == 5.5:
            text = (label + "\n" +
                    r"$\binom{M+1}{2}$ coefficient fields")
        elif level == 8.0:
            text = (label + "\n" +
                    r"$\binom{M+N-1}{N}$ coefficient fields")
        else:
            text = label
        ax.text(br[0] + 0.95, br[1] + 0.12, text, fontsize=10.5,
                ha="left", va="center",
                color="#0a7d28" if level >= 5.5 else "black")

    # Omitted first-order directions and derivative orders.
    for level, color in ((FIRST_ORDER_DOT_LEVEL, "#245b85"),
                         (ORDER_DOT_LEVEL, "0.4")):
        cx, cy = corner(0.5, 0.5, level * DZ, x0)
        ax.text(cx, cy, r"$\vdots$", fontsize=24, color=color,
                ha="center", va="center", zorder=8)

    # The 2x2 board is an interior slice, so every grey edge is exchanged.
    fig.text(0.5, 0.125,
             "interior slice of a larger decomposition:  "
             "grey = exchanged ghost cells; outward arrows continue to neighbouring ranks",
             fontsize=10,
             ha="center", color="0.25")

    # exchange-direction legend (colours match the arrows)
    fig.text(0.30, 0.055,
             r"$\mathbf{N\!\leftrightarrow\!S}$  row halo — contiguous "
             "(count = ny of MPI_OTINUM)", fontsize=10.5, ha="center",
             color=NS_COLOR)
    fig.text(0.74, 0.055,
             r"$\mathbf{E\!\leftrightarrow\!W}$  column halo — strided "
             "(MPI_Type_vector)", fontsize=10.5, ha="center", color=EW_COLOR)

    # the unifying note: stack depth = the jet
    bx, by = corner(0.0, 0.0, 0.0, x0)
    ax.annotate("one ghost cell ="
                "\none complete " r"$\mathrm{OTI}_M^N$ jet"
                "\n(one MPI_OTINUM)",
                xy=(corner(0.0, 0.0, top_z, x0)[0],
                    corner(0.0, 0.0, top_z / 2, x0)[1]),
                xytext=(bx - 4.45, by + top_z * 0.48),
                fontsize=10.5, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="none", alpha=0.96),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4,
                                connectionstyle="arc3,rad=0.12"),
                zorder=9)

    ax.set_xlim(corner(0, 0, 0, x0)[0] - 4.8,
                corner(1, 1, 0, x0)[0] + 4.2)
    ax.set_ylim(-1.0, top_z + DY + 0.5)
    fig.text(0.5, 0.95,
             r"Halo exchange moves the complete $\mathrm{OTI}_M^N$ tower",
             fontsize=14, fontweight="bold", ha="center")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
