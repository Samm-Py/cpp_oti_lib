#!/usr/bin/env python3
"""Conceptual views of a distributed OTI_M^N evaluation.

oti_decompose.png : an ordinary field on a grid (no rank slices) -> the seeded
        input jet, decomposed across ranks. Shows the "lift into OTI + split
        across ranks" step. Its RHS is the LHS of the next figure.
oti_jet_slices.png : the seeded input jet (left) -> the output jet from one
        evaluation (right), one colored plane per total-degree group. Shows the
        "each rank evaluates" step for a generic otinum<M,N>.

The planes group coefficients by total degree |alpha|. Order k contains
binom(M+k-1, k) coefficient fields; the complete jet contains
binom(M+N, N). Horizontal ellipses denote omitted directions; vertical dots
denote omitted orders. Only the value and M first-order directions are seeded on
input. Orders 2 through N start at zero and emerge from the evaluation.

Usage: plot_jet_slices.py [jet_slices.png [decompose.png]]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

# oblique (cavalier) projection: local (u, v) in [0,1]^2 at stack height z maps to
#   screen = (x0 + SX*u + DX*v,  z + DY*v)
SX = 2.65            # plane width on screen
DX, DY = 1.1, 0.82   # depth vector (how far "back" goes up-and-right)
DZ = 1.12            # vertical spacing between stacked layers
FIGURE_DPI = 220      # publication-quality raster output for the docs

FIRST_ORDER_COLOR = "#73A9D8"
DOMAIN_COLOR = "#F3C969"
RANKS = 4
ORDER_LAYERS = [
    (0, r"$|\alpha|=0:\quad c_{\mathbf{0}}=f$"
        "\n" r"$1$ coefficient field", DOMAIN_COLOR),
    (2.5, r"$|\alpha|=1:\quad \{c_\alpha\}$"
        "\n" r"$M$ coefficient fields", FIRST_ORDER_COLOR),
    (5.5, r"$|\alpha|=2:\quad \{c_\alpha\}$"
        "\n" r"$\binom{M+1}{2}$ coefficient fields", "#9BCB8C"),
    (8, r"$|\alpha|=N:\quad \{c_\alpha\}$"
        "\n" r"$\binom{M+N-1}{N}$ coefficient fields", "#70C1B3"),
]
FIRST_ORDER_PLANES = [
    (1, r"$e_0$"),
    (2, r"$e_1$"),
    (4, r"$e_{M-1}$"),
]
FIRST_ORDER_OUTPUT_PLANES = [
    (1, r"$c_{e_0}$"),
    (2, r"$c_{e_1}$"),
    (4, r"$c_{e_{M-1}}$"),
]
FIRST_ORDER_DOT_LEVEL = 3
ORDER_DOT_LEVEL = 6.75


def corner(u, v, z, x0):
    return (x0 + SX * u + DX * v, z + DY * v)


def plane_polygon(z, x0):
    return [corner(0, 0, z, x0), corner(1, 0, z, x0),
            corner(1, 1, z, x0), corner(0, 1, z, x0)]


def projection_lines(ax, x0, z_bot, z_top, color="0.6"):
    for (u, v) in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        xb, yb = corner(u, v, z_bot, x0)
        xt, yt = corner(u, v, z_top, x0)
        ax.plot([xb, xt], [yb, yt], ls=(0, (4, 3)), color=color, lw=0.9,
                zorder=1)


def draw_cell_grid(ax, z, x0):
    """Draw the faint conceptual cell grid shared by input and output planes."""
    for t in np.linspace(0, 1, 9):
        a0, a1 = corner(t, 0, z, x0), corner(t, 1, z, x0)
        b0, b1 = corner(0, t, z, x0), corner(1, t, z, x0)
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="white", lw=0.45,
                alpha=0.35, zorder=2.5)
        ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color="white", lw=0.45,
                alpha=0.35, zorder=2.5)


def draw_coefficient_plane(ax, color, z, x0):
    poly = plane_polygon(z, x0)
    ax.add_patch(Polygon(poly, closed=True, facecolor=color, alpha=0.72,
                         edgecolor="none", zorder=2))
    # A faint grid and strong outline match the visual language of mpi_halo.png.
    draw_cell_grid(ax, z, x0)
    ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="black",
                         lw=1.8, zorder=3))


def draw_flat_plane(ax, z, x0, facecolor, edge="0.35", grid=False, alpha=1.0):
    poly = plane_polygon(z, x0)
    ax.add_patch(Polygon(poly, closed=True, facecolor=facecolor, alpha=alpha,
                         edgecolor=edge, lw=1.5, zorder=2))
    if grid:
        for t in np.linspace(0, 1, 5):
            a0, a1 = corner(t, 0, z, x0), corner(t, 1, z, x0)
            b0, b1 = corner(0, t, z, x0), corner(1, t, z, x0)
            ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="0.72", lw=0.7,
                    zorder=2.5)
            ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color="0.72", lw=0.7,
                    zorder=2.5)


def draw_rank_slices(ax, z, x0, color="black", lw=1.6):
    """Draw the 2x2 rank boundaries used in the halo diagram."""
    for axis in ("u", "v"):
        if axis == "u":
            a, b = corner(0.5, 0, z, x0), corner(0.5, 1, z, x0)
        else:
            a, b = corner(0, 0.5, z, x0), corner(1, 0.5, z, x0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=3.5)


def draw_decomposed_domain(ax, z, x0):
    """The bottom input plane, split into four square rank blocks."""
    rank_cells = [
        (0, 0, 0), (0, 1, 1),
        (1, 0, 2), (1, 1, 3),
    ]
    for uh, vh, r in rank_cells:
        u0, u1 = uh * 0.5, (uh + 1) * 0.5
        v0, v1 = vh * 0.5, (vh + 1) * 0.5
        sub = [corner(u0, v0, z, x0), corner(u1, v0, z, x0),
               corner(u1, v1, z, x0), corner(u0, v1, z, x0)]
        ax.add_patch(Polygon(sub, closed=True, facecolor=DOMAIN_COLOR,
                             alpha=0.72, edgecolor="black", lw=1.6, zorder=2))
        cx, cy = corner((u0 + u1) / 2, (v0 + v1) / 2, z, x0)
        ax.text(cx, cy, f"{r}", fontsize=10, fontweight="bold",
                color="0.1", ha="center", va="center", zorder=4)
    draw_cell_grid(ax, z, x0)
    draw_rank_slices(ax, z, x0)
    ax.add_patch(Polygon(plane_polygon(z, x0), closed=True, fill=False,
                         edgecolor="black", lw=1.8, zorder=3))


def draw_plain_grid(ax, x0, z):
    """A single ordinary-field plane: a grid with NO rank slices."""
    draw_flat_plane(ax, z, x0, facecolor=DOMAIN_COLOR, edge="black", grid=True,
                    alpha=0.72)
    ax.add_patch(Polygon(plane_polygon(z, x0), closed=True, fill=False,
                         edgecolor="black", lw=1.8, zorder=4))


def draw_vertical_dots(ax, x0, z, color="0.4"):
    """Mark omitted total-degree groups between the displayed planes."""
    c = corner(0.5, 0.5, z, x0)
    ax.text(c[0], c[1], r"$\vdots$", fontsize=24, color=color,
            ha="center", va="center", zorder=8)


def draw_input_tower(ax, x0, layers, labels_side="left",
                     caption="seeded input jet,\ndecomposed across ranks",
                     decomposed=True):
    """Generic input jet grouped by total degree.

    Each displayed first-order direction owns a separate plane; vertical dots
    stand for e_2 ... e_{M-2}. Higher-order groups are drawn empty/dashed because
    they are zero before evaluation. The upper dots stand for orders 3 ... N-1."""
    top = layers[-1][0] * DZ
    projection_lines(ax, x0, 0.0, top, color="0.7")

    if decomposed:
        draw_decomposed_domain(ax, 0.0, x0)
    else:
        draw_flat_plane(ax, 0.0, x0, facecolor=DOMAIN_COLOR, edge="black",
                        alpha=0.72)
        draw_cell_grid(ax, 0.0, x0)
        ax.add_patch(Polygon(plane_polygon(0.0, x0), closed=True, fill=False,
                             edgecolor="black", lw=1.8, zorder=4))

    for level, epsilon in FIRST_ORDER_PLANES:
        z = level * DZ
        draw_flat_plane(ax, z, x0, facecolor=FIRST_ORDER_COLOR, edge="black",
                        alpha=0.72)
        draw_cell_grid(ax, z, x0)
        if decomposed:
            draw_rank_slices(ax, z, x0)
        ax.add_patch(Polygon(plane_polygon(z, x0), closed=True, fill=False,
                             edgecolor="black", lw=1.8, zorder=4))
        c = corner(0.5, 0.5, z, x0)
        ax.text(c[0], c[1], epsilon, fontsize=12, color="#245b85",
                fontweight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.62))
    draw_vertical_dots(ax, x0, FIRST_ORDER_DOT_LEVEL * DZ, color="#245b85")

    for level, _, _ in layers[2:]:
        z = level * DZ
        poly = plane_polygon(z, x0)
        ax.add_patch(Polygon(poly, closed=True, facecolor="white", alpha=0.5,
                             edgecolor="none", zorder=2))
        if decomposed:
            draw_rank_slices(ax, z, x0, color="0.8", lw=1.0)
        ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="0.6",
                             lw=1.4, ls=(0, (4, 3)), zorder=4))
        c = corner(0.5, 0.5, z, x0)
        ax.text(c[0], c[1], "$0$", fontsize=12.5, color="0.55", ha="center",
                va="center", zorder=5)
    draw_vertical_dots(ax, x0, ORDER_DOT_LEVEL * DZ, color="0.55")

    dom_label = ("physical field,\nsplit across ranks" if decomposed
                 else "physical field")
    seeded = [(0.0, dom_label, "black")]
    if labels_side == "left":
        for z, txt, col in seeded:
            if z == 0.0:
                # point to the FRONT-left of the domain and sit low-left, so the
                # label clears the e_0 plane stacked just above it
                front = corner(0.0, 0.0, z, x0)
                tx, ty = front[0] - 0.30, front[1] + 0.30
                anchor = (front[0] - 0.03, front[1] + 0.05)
            else:
                bl = corner(0.0, 1.0, z, x0)
                tx, ty = bl[0] - 0.9, bl[1] + 0.12
                anchor = (bl[0] - 0.05, bl[1] + 0.02)
            ax.annotate("", xy=anchor, xytext=(tx, ty), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
            ax.text(tx, ty, txt, fontsize=11 if z == 0.0 else 11.5,
                    ha="right", va="center", color=col,
                    fontweight="bold" if col != "black" else "normal")
        bl = corner(0.0, 1.0, 2.5 * DZ, x0)
        ax.annotate("", xy=(bl[0] - 0.05, bl[1]),
                    xytext=(bl[0] - 1.05, bl[1]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#527fa6", lw=1.2))
        ax.text(bl[0] - 1.05, bl[1],
                r"$M$ seeded first-order" "\n" "directions",
                fontsize=10.5, ha="right", va="center",
                color="#527fa6", fontweight="bold")
        bl = corner(0.0, 1.0, layers[2][0] * DZ, x0)
        ax.annotate("", xy=(bl[0] - 0.05, bl[1] + 0.02),
                    xytext=(bl[0] - 1.05, bl[1] + 0.2), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.1))
        ax.text(bl[0] - 1.05, bl[1] + 0.2,
                r"orders $2,\ldots,N$:" "\n" r"not seeded ($=0$)",
                fontsize=10.5, ha="right", va="center", color="0.45",
                style="italic")
    elif labels_side == "right":
        for z, txt, col in seeded:
            br = corner(1.0, 1.0, z, x0)
            tx = br[0] + (1.0 if z == 0.0 else 0.8)
            ty = br[1] + (0.30 if z == 0.0 else 0.12)
            ax.annotate("", xy=(br[0] + 0.05, br[1] + 0.02), xytext=(tx, ty),
                        zorder=6, arrowprops=dict(arrowstyle="-|>", color="0.4",
                                                  lw=1.2))
            ax.text(tx, ty, txt, fontsize=10.5 if z == 0.0 else 11,
                    ha="left", va="center", color=col,
                    fontweight="bold" if col != "black" else "normal")
        br = corner(1.0, 1.0, 2.5 * DZ, x0)
        ax.annotate("", xy=(br[0] + 0.05, br[1]),
                    xytext=(br[0] + 1.0, br[1]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#527fa6", lw=1.2))
        ax.text(br[0] + 1.0, br[1],
                r"$M$ seeded first-order" "\n" "directions",
                fontsize=10, ha="left", va="center",
                color="#527fa6", fontweight="bold")
        br = corner(1.0, 1.0, layers[2][0] * DZ, x0)
        ax.annotate("", xy=(br[0] + 0.05, br[1] + 0.02),
                    xytext=(br[0] + 1.0, br[1] + 0.2), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.1))
        ax.text(br[0] + 1.0, br[1] + 0.2,
                r"orders $2,\ldots,N$:" "\n" r"not seeded ($=0$)",
                fontsize=10, ha="left", va="center", color="0.45",
                style="italic")

    if caption:
        ax.text(corner(0.5, 0, 0, x0)[0], corner(0.5, 0, 0, x0)[1] - 0.45,
                caption, fontsize=11, ha="center", va="top")


def draw_output_tower(ax, x0, layers):
    """Generic output jet: one plane per displayed total-degree group."""
    top = layers[-1][0] * DZ
    projection_lines(ax, x0, 0.0, top)
    for level, label, color in layers:
        if label.startswith(r"$|\alpha|=1"):
            for first_level, coefficient in FIRST_ORDER_OUTPUT_PLANES:
                z = first_level * DZ
                draw_coefficient_plane(ax, color, z, x0)
                draw_rank_slices(ax, z, x0)
                c = corner(0.5, 0.5, z, x0)
                ax.text(c[0], c[1], coefficient, fontsize=11.5,
                        color="#245b85", fontweight="bold",
                        ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.62))
            draw_vertical_dots(ax, x0, FIRST_ORDER_DOT_LEVEL * DZ,
                               color="#245b85")
            br = corner(1.0, 1.0, FIRST_ORDER_OUTPUT_PLANES[-1][0] * DZ, x0)
            lx, ly = br[0] + 0.9, br[1] + 0.18
            ax.annotate("", xy=(lx - 0.05, ly - 0.05), xytext=br, zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3))
            ax.text(lx, ly, label, fontsize=10.8, ha="left", va="center",
                    color="black")
            continue
        z = level * DZ
        draw_coefficient_plane(ax, color, z, x0)
        draw_rank_slices(ax, z, x0)   # each rank's output slice
        if level == 0:
            rank_centres = [
                (0.25, 0.25, 0), (0.25, 0.75, 1),
                (0.75, 0.25, 2), (0.75, 0.75, 3),
            ]
            for u, v, r in rank_centres:
                cx, cy = corner(u, v, z, x0)
                ax.text(cx, cy, f"{r}", fontsize=9.5, fontweight="bold",
                        color="black", ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.14", fc="white",
                                  ec="none", alpha=0.65))
        br = corner(1.0, 1.0, z, x0)
        lx, ly = br[0] + 0.9, br[1] + 0.18
        ax.annotate("", xy=(lx - 0.05, ly - 0.05), xytext=br, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3))
        col = "#0a7d28" if level >= 2 else "black"
        ax.text(lx, ly, label, fontsize=10.8, ha="left", va="center", color=col)
    draw_vertical_dots(ax, x0, ORDER_DOT_LEVEL * DZ)
    ax.text(corner(1.0, 1.0, 0, x0)[0] + 0.9,
            corner(1.0, 1.0, 0, x0)[1] - 0.50,
            r"total: $\binom{M+N}{N}$ coefficient fields",
            fontsize=10.5, color="0.2", ha="left", va="center")
    ax.text(corner(0.5, 0, 0, x0)[0], corner(0.5, 0, 0, x0)[1] - 0.45,
            r"output jet $\mathrm{OTI}_M^N$:"
            "\nall coefficient groups",
            fontsize=11, ha="center", va="top")


def main(out_path="oti_jet_slices.png"):
    """The evaluate figure: seeded input jet -> output coefficient tower."""
    layers = ORDER_LAYERS
    fig, ax = plt.subplots(figsize=(14.5, 9.5))
    ax.set_aspect("equal")
    ax.axis("off")

    xL = -6.0
    draw_input_tower(ax, xL, layers, labels_side="left")

    # ---- evaluate arrow (extra gap so the label clears both stacks) --------
    x0 = 2.2
    arrow_z = 3.1 * DZ
    a_start = corner(1.06, 0.5, arrow_z, xL)
    a_end = (corner(0, 0.5, arrow_z, x0)[0] - 0.3,
             corner(0, 0.5, arrow_z, x0)[1])
    ax.add_patch(FancyArrowPatch(a_start, a_end, connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=24, lw=2.4,
                                 color="#222222", zorder=7))
    tx = (a_start[0] + a_end[0]) / 2
    ax.text(tx + 0.2, a_start[1] + 0.86,
            "each rank evaluates\n" + r"$F(\mathbf{x}^\ast)$ on its own block",
            fontsize=10.2, fontweight="bold", ha="center", va="center",
            linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="none", alpha=0.96),
            zorder=9)

    draw_output_tower(ax, x0, layers)

    top = layers[-1][0] * DZ
    ax.set_xlim(corner(0, 1, 0, xL)[0] - 3.8, corner(1, 1, 0, x0)[0] + 4.2)
    ax.set_ylim(-1.15, top + DY + 0.48)
    fig.text(0.46, 0.95, r"A generic $\mathrm{OTI}_M^N$ jet: "
             "seeded directions in  →  all derivative orders out",
             fontsize=13.5, fontweight="bold",
             ha="center")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out_path)
    plt.close(fig)


def decompose_image(out_path="oti_decompose.png"):
    """The setup figure, in order: ordinary field -> lift into OTI (whole jet on
    root) -> scatter/broadcast across ranks (rank-decomposed jet)."""
    layers = ORDER_LAYERS
    fig, ax = plt.subplots(figsize=(20.5, 9.5))
    ax.set_aspect("equal")
    ax.axis("off")
    zG = 3.2 * DZ   # height of the connecting arrows / the plain grid

    def stage_arrow(xa_right_tower, xb_tower, bold, italic):
        a_start = corner(1.05, 0.5, zG, xa_right_tower)
        a_end = (corner(0, 0.5, zG, xb_tower)[0] - 0.32,
                 corner(0, 0.5, zG, xb_tower)[1])
        ax.add_patch(FancyArrowPatch(a_start, a_end,
                                     connectionstyle="arc3,rad=-0.07",
                                     arrowstyle="-|>", mutation_scale=22, lw=2.2,
                                     color="#222222", zorder=7))
        midx = (a_start[0] + a_end[0]) / 2
        ax.text(midx + 0.1, a_start[1] + 1.02,
                bold + "\n" + italic, fontsize=10.2,
                fontweight="bold", ha="center", va="center",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="none", alpha=0.96),
                zorder=9)

    # stage 1: an ordinary field on a plain grid (no rank slices)
    xG = -14.8
    draw_plain_grid(ax, xG, zG)
    ax.text(corner(0.5, 0.0, zG, xG)[0], corner(0.5, 0.0, zG, xG)[1] - 0.45,
            "an ordinary field\non a grid", fontsize=11, ha="center", va="top")

    # arrow 1: lift into OTI (pure on-root operation, no MPI yet)
    xT1 = -7.8
    stage_arrow(xG, xT1, "lift into OTI",
                r"(seed $e_0,\ldots,e_{M-1}$)")

    # stage 2: the lifted whole seeded jet, still on root (NOT decomposed)
    draw_input_tower(ax, xT1, layers, labels_side=None,
                     caption="lifted jet field\n(on root)", decomposed=False)

    # arrow 2: scatter / broadcast across ranks (the MPI movement)
    xT2 = 0.2
    stage_arrow(xT1, xT2, "scatter / broadcast", "across ranks")

    # stage 3: the rank-decomposed seeded jet (labels on the right)
    draw_input_tower(ax, xT2, layers, labels_side="right",
                     caption="seeded input jet,\ndecomposed across ranks",
                     decomposed=True)

    top = layers[-1][0] * DZ
    ax.set_xlim(corner(0, 1, zG, xG)[0] - 1.8, corner(1, 1, 0, xT2)[0] + 4.4)
    ax.set_ylim(-1.15, top + DY + 0.48)
    fig.text(0.5, 0.96, "Setup: lift the field into OTI on root, then scatter it "
             "across ranks", fontsize=13.5, fontweight="bold", ha="center")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out_path)
    plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) >= 1 else "oti_jet_slices.png")
    decompose_image(a[1] if len(a) >= 2 else "oti_decompose.png")
