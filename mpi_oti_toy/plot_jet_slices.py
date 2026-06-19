#!/usr/bin/env python3
"""Conceptual view of a distributed OTI evaluation.

LEFT  : the seeded input jet, drawn as a stack of layers -- the base domain plus
        the two nilpotent (hypercomplex) perturbation directions e_0 and e_1.
        Every grid point is lifted x -> x + e_0, y -> y + e_1. Only first-order
        perturbations are seeded.
RIGHT : the output jet from a single evaluation of a general f -- one colored
        plane per coefficient (the value and every derivative field), with
        dashed projection lines tying the stack to one footprint. The colors
        distinguish coefficient layers; they do not encode function values.

Slices are the coefficients otinum<2,2> actually stores:
  c_00 = f            c_10 = f_x       c_01 = f_y
  c_20 = f_xx / 2     c_11 = f_xy      c_02 = f_yy / 2

Usage: plot_jet_slices.py [out.png]
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

E0_COLOR = "#73A9D8"
E1_COLOR = "#E08E79"
DOMAIN_COLOR = "#F3C969"
RANKS = 4
COEFFICIENTS = [
    (r"$c_{00}=f$", DOMAIN_COLOR),
    (r"$c_{10}=\partial_x f$", "#73A9D8"),
    (r"$c_{01}=\partial_y f$", "#E08E79"),
    (r"$c_{20}=\frac{1}{2}\partial_{xx} f$", "#9BCB8C"),
    (r"$c_{11}=\partial_{xy} f$", "#B59AD2"),
    (r"$c_{02}=\frac{1}{2}\partial_{yy} f$", "#70C1B3"),
]


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


def main(out_path="oti_jet_slices.png"):
    coeffs = COEFFICIENTS

    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- LEFT: the seeded input jet as a stack -----------------------------
    xL = -5.15
    projection_lines(ax, xL, 0.0, 2 * DZ, color="0.7")
    # layer 0: the domain, decomposed across MPI ranks
    draw_decomposed_domain(ax, 0.0, xL)
    # layer 1: e_0 perturbation direction (perturb x), sliced across ranks
    draw_flat_plane(ax, DZ, xL, facecolor=E0_COLOR, edge="black", alpha=0.72)
    draw_cell_grid(ax, DZ, xL)
    draw_rank_slices(ax, DZ, xL)
    ax.add_patch(Polygon(plane_polygon(DZ, xL), closed=True, fill=False,
                         edgecolor="black", lw=1.8, zorder=4))
    # layer 2: e_1 perturbation direction (perturb y), sliced across ranks
    draw_flat_plane(ax, 2 * DZ, xL, facecolor=E1_COLOR, edge="black", alpha=0.72)
    draw_cell_grid(ax, 2 * DZ, xL)
    draw_rank_slices(ax, 2 * DZ, xL)
    ax.add_patch(Polygon(plane_polygon(2 * DZ, xL), closed=True, fill=False,
                         edgecolor="black", lw=1.8, zorder=4))

    # left-side labels for the input layers
    for z, txt, col in [(0.0, "domain $(x, y)$,\nsplit across ranks", "black"),
                        (DZ, r"$+\,e_0$  (seed $\partial_x$)", E0_COLOR),
                        (2 * DZ, r"$+\,e_1$  (seed $\partial_y$)", E1_COLOR)]:
        bl = corner(0.0, 1.0, z, xL)
        tx = bl[0] - (1.25 if z == 0.0 else 0.9)
        ty = bl[1] + (0.32 if z == 0.0 else 0.12)
        ax.annotate("", xy=(bl[0] - 0.05, bl[1] + 0.02), xytext=(tx, ty),
                    zorder=6, arrowprops=dict(arrowstyle="-|>", color="0.4",
                                              lw=1.2))
        ax.text(tx, ty, txt, fontsize=11 if z == 0.0 else 11.5,
                ha="right", va="center", color=col,
                fontweight="bold" if col != "black" else "normal")
    ax.text(corner(0.5, 0, 0, xL)[0], corner(0.5, 0, 0, xL)[1] - 0.45,
            "seeded input jet,\ndecomposed across ranks",
            fontsize=11, ha="center", va="top")

    # ---- evaluate arrow (extra gap so the label clears both stacks) --------
    x0 = 1.35                                 # output tower origin, shifted right
    a_start = corner(1.06, 0.5, DZ, xL)
    a_end = (corner(0, 0.5, DZ, x0)[0] - 0.3, corner(0, 0.5, DZ, x0)[1])
    ax.add_patch(FancyArrowPatch(a_start, a_end, connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=24, lw=2.4,
                                 color="#222222", zorder=7))
    tx = (a_start[0] + a_end[0]) / 2
    ax.text(tx + 0.2, a_start[1] + 1.18, "each rank evaluates", fontsize=10.5,
            fontweight="bold", ha="center")
    ax.text(tx + 0.2, a_start[1] + 0.80,
            r"$f(x^\ast,y^\ast)$ on its block", fontsize=9.5,
            ha="center", style="italic")

    # ---- RIGHT: the output coefficient tower -------------------------------
    top = (len(coeffs) - 1) * DZ
    projection_lines(ax, x0, 0.0, top)
    for k, (label, color) in enumerate(coeffs):
        z = k * DZ
        draw_coefficient_plane(ax, color, z, x0)
        draw_rank_slices(ax, z, x0)   # each rank's output slice
        if k == 0:
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
        col = "#0a7d28" if k >= 3 else "black"   # 2nd-order emerge -> highlight
        ax.text(lx, ly, label, fontsize=12.5, ha="left", va="center", color=col)
    ax.text(corner(1.0, 1.0, 3.5 * DZ, x0)[0] + 1.5,
            corner(1.0, 1.0, 5 * DZ, x0)[1] + 1.05,
            "second-order:\nemerge from the\nevaluation",
            fontsize=9.5, color="#0a7d28", ha="left", va="center", style="italic")
    ax.text(corner(0.5, 0, 0, x0)[0], corner(0.5, 0, 0, x0)[1] - 0.45,
            "output jet:\nall coefficient fields",
            fontsize=11, ha="center", va="top")

    ax.set_xlim(corner(0, 1, 0, xL)[0] - 2.7, corner(1, 1, 0, x0)[0] + 2.1)
    ax.set_ylim(-1.15, top + DY + 0.48)
    fig.text(0.46, 0.95, "An OTI jet: seeded perturbations in  →  the full "
             "derivative tower out", fontsize=13.5, fontweight="bold",
             ha="center")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
