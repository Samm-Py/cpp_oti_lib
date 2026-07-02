#!/usr/bin/env python3
"""Plot full-grid absolute errors between OTI and finite differences.

Usage:
    plot_fd_error.py INPUT.csv [OUTPUT.png]
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def main(csv_path, out_path="mpi_halo_fd_error.png"):
    rows = []
    with open(csv_path, newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append((int(row["i"]), int(row["j"]),
                         float(row["error_west"]), float(row["error_south"])))

    n = max(max(i, j) for i, j, _, _ in rows)
    west = np.zeros((n, n))
    south = np.zeros((n, n))
    for i, j, ew, es in rows:
        west[i - 1, j - 1] = ew
        south[i - 1, j - 1] = es

    positive = np.concatenate((west[west > 0], south[south > 0]))
    vmin = max(positive.min(), 1.0e-16)
    vmax = max(west.max(), south.max())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    images = []
    for ax, field, title in zip(
            axes, (west, south),
            (r"$|\mathrm{OTI}-\mathrm{FD}|$ for $\partial u/\partial T_\mathrm{west}$",
             r"$|\mathrm{OTI}-\mathrm{FD}|$ for $\partial u/\partial T_\mathrm{south}$")):
        image = ax.imshow(field.T, origin="lower", cmap="magma",
                          norm=LogNorm(vmin=vmin, vmax=vmax),
                          extent=(1, n, 1, n), interpolation="nearest")
        images.append(image)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("grid index $i$")
        ax.set_ylabel("grid index $j$")
        ax.set_aspect("equal")

    cbar = fig.colorbar(images[0], ax=axes, shrink=0.88, pad=0.03)
    cbar.set_label("absolute error")
    fig.suptitle(r"Centered finite difference ($h=10^{-6}$) versus OTI",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
