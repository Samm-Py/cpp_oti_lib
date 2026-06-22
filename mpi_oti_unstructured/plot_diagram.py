#!/usr/bin/env python3
"""Schematic for the unstructured ghost-exchange (MPI_Type_indexed) tutorial.

Two panels, sharing one small illustrative graph (24 nodes, ring + a few chords,
partitioned into 4 contiguous blocks -- the real example uses 240 nodes):

  LEFT  -- the graph on a circle, nodes coloured by owning rank (contiguous arcs).
           Ring edges run along the circle; chords cross between ranks. The chords
           that leave rank 0 for rank 2 are highlighted: those are the edges that
           create scattered ghosts.
  RIGHT -- rank 0's local array: owned slots, then ghost slots grouped by owner.
           Receiving is a contiguous count per owner; SENDING the owned nodes a
           neighbour needs is a scattered subset -> one MPI_Type_indexed.

Usage: plot_diagram.py [out.png]
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

V = 24
PARTS = 4
PER = V // PARTS                         # 6 owned nodes per rank
RANK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
SRC_A, SRC_B = 0, V - 1
FOCUS = 0                                # the rank whose local view we draw
SEND_TO = 2                              # the neighbour whose send we illustrate
FIGURE_DPI = 200

# value + first-order coefficient colours (match the reduce-tree figure)
JET_COLORS = ["#F3C969", "#73A9D8", "#E08E79"]

# Ring edges are implicit (g, g+1). These chords are hand-picked to be legible and
# to give rank 0 a scattered send list to each neighbour.
CHORDS = [(1, 14), (3, 16), (2, 9), (4, 19), (8, 20), (13, 22)]


def owner(g):
    return g // PER


def adjacency():
    adj = {g: set() for g in range(V)}
    for i in range(V):                    # ring
        j = (i + 1) % V
        adj[i].add(j); adj[j].add(i)
    for a, b in CHORDS:
        adj[a].add(b); adj[b].add(a)
    return {g: sorted(v) for g, v in adj.items()}


def node_xy(g):
    ang = np.pi / 2 - 2 * np.pi * g / V   # node 0 at top, clockwise
    return np.cos(ang), np.sin(ang)


def draw_graph(ax, adj):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("An unstructured graph, partitioned into contiguous blocks",
                 fontsize=12.5, fontweight="bold", pad=10)

    # edges: ring faint grey, chords darker; chords FOCUS->SEND_TO highlighted
    for g in range(V):
        for h in adj[g]:
            if h <= g:
                continue
            x0, y0 = node_xy(g); x1, y1 = node_xy(h)
            is_ring = (h == (g + 1) % V) or (g == (h + 1) % V)
            hot = ({owner(g), owner(h)} == {FOCUS, SEND_TO})
            if hot:
                ax.plot([x0, x1], [y0, y1], color="#C44E52", lw=2.6,
                        zorder=3, alpha=0.95)
            elif is_ring:
                ax.plot([x0, x1], [y0, y1], color="0.78", lw=1.4, zorder=1)
            else:
                ax.plot([x0, x1], [y0, y1], color="0.55", lw=1.0,
                        zorder=1, alpha=0.7)

    # nodes coloured by owning rank
    for g in range(V):
        x, y = node_xy(g)
        ax.add_patch(Circle((x, y), 0.062, facecolor=RANK_COLORS[owner(g)],
                            edgecolor="black", lw=1.2, zorder=4))
        if g in (SRC_A, SRC_B):           # sources get a bold outline + star
            ax.add_patch(Circle((x, y), 0.092, facecolor="none",
                                edgecolor="black", lw=2.2, zorder=5))
            ax.text(x, y, "$\\star$", fontsize=11, ha="center", va="center",
                    color="white", zorder=6)
        lx, ly = node_xy(g)
        ax.text(lx * 1.16, ly * 1.16, str(g), fontsize=8, ha="center",
                va="center", color="0.3", zorder=4)

    # rank-arc labels
    for r in range(PARTS):
        gmid = r * PER + (PER - 1) / 2
        x, y = node_xy(gmid)
        ax.text(x * 1.42, y * 1.42, f"rank {r}", fontsize=10.5,
                fontweight="bold", color=RANK_COLORS[r], ha="center",
                va="center")

    ax.text(0, -1.62,
            "sources $\\star$ (nodes %d, %d) seeded $A=e_0$, $B=e_1$;  "
            "red chords = rank 0$\\to$rank 2 cross-edges" % (SRC_A, SRC_B),
            fontsize=9, ha="center", color="0.3")
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.8, 1.7)


def draw_local_array(ax, adj):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Rank 0's local array: owned slots + ghost slots",
                 fontsize=12.5, fontweight="bold", pad=10)

    owned = list(range(FOCUS * PER, FOCUS * PER + PER))
    # ghosts: remote endpoints of edges touching owned, grouped by owner, sorted
    ghosts_by_owner = {}
    for u in owned:
        for v in adj[u]:
            if owner(v) != FOCUS:
                ghosts_by_owner.setdefault(owner(v), set()).add(v)
    ghost_order = []
    ghost_groups = []                      # (rank, [global ids], slot range)
    for r in sorted(ghosts_by_owner):
        ids = sorted(ghosts_by_owner[r])
        start = len(owned) + len(ghost_order)
        ghost_order += ids
        ghost_groups.append((r, ids, (start, start + len(ids))))

    slots = owned + ghost_order
    n = len(slots)
    W, H, y0 = 0.84, 0.84, 0.0

    def slot_color(idx):
        g = slots[idx]
        return RANK_COLORS[owner(g)]

    for idx in range(n):
        x = idx
        ax.add_patch(Rectangle((x, y0), W, H, facecolor=slot_color(idx),
                               edgecolor="black", lw=1.3, zorder=3,
                               alpha=0.9))
        ax.text(x + W / 2, y0 + H / 2, str(slots[idx]), fontsize=9.5,
                ha="center", va="center", fontweight="bold", color="white",
                zorder=4)
        ax.text(x + W / 2, y0 - 0.22, str(idx), fontsize=7.5, ha="center",
                va="center", color="0.5", zorder=4)   # slot index

    # brackets: owned region, and each ghost group (contiguous = recv count)
    def bracket(a, b, label, color, yb):
        ax.plot([a, a, b, b], [yb + 0.06, yb, yb, yb + 0.06], color=color,
                lw=1.6)
        ax.text((a + b) / 2, yb - 0.18, label, fontsize=8.5, ha="center",
                va="top", color=color)
    bracket(0, len(owned) - 0.16, "owned (slots 0..%d)" % (len(owned) - 1),
            "0.25", y0 - 0.42)
    for r, ids, (s, e) in ghost_groups:
        bracket(s, e - 0.16,
                "ghosts from rank %d\n(recv: count=%d)" % (r, len(ids)),
                RANK_COLORS[r], y0 - 0.42)

    # the indexed SEND: scattered owned slots that SEND_TO needs
    needed = sorted({u for u in owned
                     if any(owner(v) == SEND_TO for v in adj[u])})
    disp = [owned.index(u) for u in needed]     # local slot indices
    env_y = y0 + H + 1.35
    env_x0, env_x1 = disp[0], disp[-1] + W
    env = FancyBboxPatch((env_x0 - 0.15, env_y - 0.28),
                         (env_x1 - env_x0) + 0.3, 0.78,
                         boxstyle="round,pad=0.06", fc="#fdf6e3",
                         ec=RANK_COLORS[SEND_TO], lw=1.8, zorder=4)
    ax.add_patch(env)
    ax.text((env_x0 + env_x1) / 2, env_y + 0.5,
            "one MPI_Type_indexed send to rank %d" % SEND_TO,
            fontsize=9.5, ha="center", va="bottom", fontweight="bold",
            color=RANK_COLORS[SEND_TO], zorder=5)
    ax.text((env_x0 + env_x1) / 2, env_y + 0.11,
            "displacements = %s  (block lengths all 1)" % disp,
            fontsize=9, ha="center", va="center", style="italic",
            color="0.25", zorder=5)
    for d in disp:
        ax.add_patch(Rectangle((d, y0), W, H, facecolor="none",
                               edgecolor=RANK_COLORS[SEND_TO], lw=2.6,
                               zorder=5))
        ax.add_patch(FancyArrowPatch((d + W / 2, y0 + H + 0.04),
                                     (d + W / 2, env_y - 0.28),
                                     arrowstyle="-|>", mutation_scale=11,
                                     lw=1.6, color=RANK_COLORS[SEND_TO],
                                     zorder=4))

    ax.text(n / 2.0, env_y + 1.15,
            "each slot is one jet  [ value | $\\partial A$ | $\\partial B$ ]  "
            "= one MPI_OTINUM",
            fontsize=9.5, ha="center", va="center", color="0.2")
    ax.set_xlim(-0.8, n + 0.4)
    ax.set_ylim(y0 - 1.55, env_y + 1.5)


def main(out_path="mpi_unstructured.png"):
    adj = adjacency()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 7.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.18]})
    draw_graph(axL, adj)
    draw_local_array(axR, adj)
    fig.suptitle("Unstructured ghost exchange: scattered owned nodes ship as one "
                 "MPI_Type_indexed; ghosts arrive grouped by owner",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.1)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
