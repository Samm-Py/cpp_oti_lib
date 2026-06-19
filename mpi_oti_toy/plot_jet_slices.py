#!/usr/bin/env python3
"""Visualize what an OTI jet *is*, in the style of an FE reference-element figure.

LEFT  : the seeded input jet, drawn as a stack of layers -- the base domain plus
        the two nilpotent (hypercomplex) perturbation directions e_0 and e_1.
        Every grid point is lifted x -> x + e_0, y -> y + e_1. Only first-order
        perturbations are seeded.
RIGHT : the output jet from a single evaluation of f -- one oblique plane per
        coefficient (the value and every derivative field), with dashed
        projection lines tying the stack to one footprint. The second-order
        coefficients (c_20, c_11, c_02) were never seeded; they emerge from the
        algebra during the evaluation.

We use an off-centre Gaussian bump f(x, y) = exp(-a[(x-cx)^2 + (y-cy)^2]) so all
six derivative fields are visually distinct. The OTI mechanism is identical for
any f. Slices are the coefficients otinum<2,2> actually stores:
  c_00 = f            c_10 = f_x       c_01 = f_y
  c_20 = f_xx / 2     c_11 = f_xy      c_02 = f_yy / 2

Usage: plot_jet_slices.py [out.png]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.transforms import Affine2D

# oblique (cavalier) projection: local (u, v) in [0,1]^2 at stack height z maps to
#   screen = (x0 + SX*u + DX*v,  z + DY*v)
SX = 2.3             # plane width on screen
DX, DY = 1.0, 0.78   # depth vector (how far "back" goes up-and-right)
DZ = 1.2             # vertical spacing between stacked layers

E0_COLOR = "#C44E52"
E1_COLOR = "#4C72B0"
RANKS = 4
RANK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def coefficients(n=90):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    a, cx, cy = 16.0, 0.45, 0.55
    dx, dy = X - cx, Y - cy
    g = np.exp(-a * (dx * dx + dy * dy))
    gx = -2 * a * dx * g
    gy = -2 * a * dy * g
    gxx = (-2 * a + 4 * a * a * dx * dx) * g
    gxy = (4 * a * a * dx * dy) * g
    gyy = (-2 * a + 4 * a * a * dy * dy) * g
    return [
        (r"$c_{00}=f$",                   g),
        (r"$c_{10}=\partial_x f$",        gx),
        (r"$c_{01}=\partial_y f$",        gy),
        (r"$c_{20}=\frac{1}{2}\partial_{xx} f$", 0.5 * gxx),
        (r"$c_{11}=\partial_{xy} f$",     gxy),
        (r"$c_{02}=\frac{1}{2}\partial_{yy} f$", 0.5 * gyy),
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


def draw_field_plane(ax, field, z, x0, cmap):
    m = max(abs(field.min()), abs(field.max())) or 1.0
    aff = Affine2D.from_values(SX, 0.0, DX, DY, x0, z) + ax.transData
    im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                   norm=Normalize(-m, m), transform=aff, zorder=2)
    poly = plane_polygon(z, x0)
    im.set_clip_path(Polygon(poly, closed=True, transform=ax.transData))
    ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="0.35",
                         lw=1.4, zorder=3))


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


def draw_rank_slices(ax, z, x0, color="0.45", lw=1.0):
    """Draw the rank-block boundaries (u-dividers) on a plane, so every layer of
    the jet shows the same MPI decomposition. Light grey dashed so it reads on
    both the pale input planes and the coloured output fields."""
    for r in range(1, RANKS):
        u = r / RANKS
        a, b = corner(u, 0, z, x0), corner(u, 1, z, x0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                ls=(0, (4, 3)), zorder=3.5)


def draw_decomposed_domain(ax, z, x0):
    """The bottom input plane, split into per-rank blocks (the MPI decomposition).
    Blocks are bands of the u axis (x), matching the toy's flat block partition."""
    for r in range(RANKS):
        u0, u1 = r / RANKS, (r + 1) / RANKS
        sub = [corner(u0, 0, z, x0), corner(u1, 0, z, x0),
               corner(u1, 1, z, x0), corner(u0, 1, z, x0)]
        ax.add_patch(Polygon(sub, closed=True, facecolor=RANK_COLORS[r],
                             alpha=0.45, edgecolor="none", zorder=2))
        cx, cy = corner((u0 + u1) / 2, 0.5, z, x0)
        ax.text(cx, cy, f"{r}", fontsize=10, fontweight="bold",
                color="0.1", ha="center", va="center", zorder=4)
    draw_rank_slices(ax, z, x0)
    ax.add_patch(Polygon(plane_polygon(z, x0), closed=True, fill=False,
                         edgecolor="0.3", lw=1.6, zorder=3))


def main(out_path="oti_jet_slices.png"):
    coeffs = coefficients()
    cmap = cm.RdBu_r

    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- LEFT: the seeded input jet as a stack -----------------------------
    xL = -5.0
    projection_lines(ax, xL, 0.0, 2 * DZ, color="0.7")
    # layer 0: the domain, decomposed across MPI ranks
    draw_decomposed_domain(ax, 0.0, xL)
    # layer 1: e_0 perturbation direction (perturb x), sliced across ranks
    draw_flat_plane(ax, DZ, xL, facecolor=E0_COLOR, alpha=0.22)
    draw_rank_slices(ax, DZ, xL)
    # layer 2: e_1 perturbation direction (perturb y), sliced across ranks
    draw_flat_plane(ax, 2 * DZ, xL, facecolor=E1_COLOR, alpha=0.22)
    draw_rank_slices(ax, 2 * DZ, xL)

    # left-side labels for the input layers
    for z, txt, col in [(0.0, "domain $(x, y)$,\nsplit across ranks", "black"),
                        (DZ, r"$+\,e_0$  (seed $\partial_x$)", E0_COLOR),
                        (2 * DZ, r"$+\,e_1$  (seed $\partial_y$)", E1_COLOR)]:
        bl = corner(0.0, 1.0, z, xL)
        tx, ty = bl[0] - 0.9, bl[1] + 0.12
        ax.annotate("", xy=(bl[0] - 0.05, bl[1] + 0.02), xytext=(tx, ty),
                    zorder=6, arrowprops=dict(arrowstyle="-|>", color="0.4",
                                              lw=1.2))
        ax.text(tx, ty, txt, fontsize=11.5, ha="right", va="center", color=col,
                fontweight="bold" if col != "black" else "normal")
    ax.text(corner(0.5, 0, 0, xL)[0], corner(0.5, 0, 0, xL)[1] - 0.45,
            "seeded input jet,\ndecomposed across ranks",
            fontsize=11, ha="center", va="top")

    # ---- evaluate arrow (extra gap so the label clears both stacks) --------
    x0 = 1.4                                  # output tower origin, shifted right
    a_start = corner(1.06, 0.5, DZ, xL)
    a_end = (corner(0, 0.5, DZ, x0)[0] - 0.3, corner(0, 0.5, DZ, x0)[1])
    ax.add_patch(FancyArrowPatch(a_start, a_end, connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=24, lw=2.4,
                                 color="#222222", zorder=7))
    tx = (a_start[0] + a_end[0]) / 2
    ax.text(tx, a_start[1] + 1.12, "each rank evaluates", fontsize=10.5,
            fontweight="bold", ha="center")
    ax.text(tx, a_start[1] + 0.74, r"$f(x,y)$ on its block", fontsize=9.5,
            ha="center", style="italic")

    # ---- RIGHT: the output coefficient tower -------------------------------
    top = (len(coeffs) - 1) * DZ
    projection_lines(ax, x0, 0.0, top)
    for k, (label, field) in enumerate(coeffs):
        z = k * DZ
        draw_field_plane(ax, field, z, x0, cmap)
        draw_rank_slices(ax, z, x0)   # each rank's output slice
        br = corner(1.0, 1.0, z, x0)
        lx, ly = br[0] + 0.9, br[1] + 0.18
        ax.annotate("", xy=(lx - 0.05, ly - 0.05), xytext=br, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3))
        col = "#0a7d28" if k >= 3 else "black"   # 2nd-order emerge -> highlight
        ax.text(lx, ly, label, fontsize=12.5, ha="left", va="center", color=col)
    ax.text(corner(1.0, 1.0, 3.5 * DZ, x0)[0] + 0.9,
            corner(1.0, 1.0, 5 * DZ, x0)[1] + 0.95,
            "second-order:\nemerge from the\nevaluation",
            fontsize=9.5, color="#0a7d28", ha="left", va="center", style="italic")
    ax.text(corner(0.5, 0, 0, x0)[0], corner(0.5, 0, 0, x0)[1] - 0.45,
            "output jet:\nall coefficient fields",
            fontsize=11, ha="center", va="top")

    # ---- colour key --------------------------------------------------------
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(-1, 1))
    sm.set_array([])
    cax = fig.add_axes([0.89, 0.34, 0.011, 0.34])
    cb = fig.colorbar(sm, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels([r"$-$", "0", r"$+$"])
    cb.set_label("per-slice sign / magnitude", fontsize=9)

    ax.set_xlim(corner(0, 1, 0, xL)[0] - 2.7, corner(1, 1, 0, x0)[0] + 2.2)
    ax.set_ylim(-1.4, top + DY + 0.55)
    fig.text(0.5, 0.97, "An OTI jet: seeded perturbations in  →  the full "
             "derivative tower out", fontsize=14, fontweight="bold",
             ha="center")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
