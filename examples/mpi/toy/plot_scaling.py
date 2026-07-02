#!/usr/bin/env python3
"""Plot MPI rank-scaling of the OTI grid evaluation.

Reads the CSV emitted by mpi_oti_scaling (columns: algebra,ncoeffs,ranks,
repetition,seconds) and produces a two-panel figure: parallel speedup and
parallel efficiency vs rank count, one curve per algebra.

Usage: plot_scaling.py <scaling.csv> <out.png>
"""
import csv
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main(csv_path, out_path):
    times = defaultdict(list)  # (algebra, ranks) -> [seconds]
    for row in csv.DictReader(open(csv_path)):
        times[(row["algebra"], int(row["ranks"]))].append(float(row["seconds"]))
    median = {k: st.median(v) for k, v in times.items()}

    algebras = sorted({k[0] for k in median})
    ranks = sorted({k[1] for k in median})

    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax_s.plot(ranks, ranks, "k--", lw=1, label="ideal (linear)")
    for alg in algebras:
        t1 = median[(alg, 1)]
        speedup = [t1 / median[(alg, r)] for r in ranks]
        eff = [s / r for s, r in zip(speedup, ranks)]
        ax_s.plot(ranks, speedup, "o-", label=alg)
        ax_e.plot(ranks, eff, "o-", label=alg)

    ax_s.set_xscale("log", base=2)
    ax_s.set_yscale("log", base=2)
    ax_s.set_xticks(ranks)
    ax_s.set_yticks(ranks)
    ax_s.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_s.get_yaxis().set_major_formatter(ScalarFormatter())
    ax_s.set_xlabel("MPI ranks")
    ax_s.set_ylabel("speedup  (T$_1$ / T$_n$)")
    ax_s.set_title("Parallel speedup")
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(fontsize=8)

    ax_e.axhline(1.0, color="k", ls="--", lw=1, label="ideal (100%)")
    ax_e.set_xscale("log", base=2)
    ax_e.set_xticks(ranks)
    ax_e.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_e.set_xlabel("MPI ranks")
    ax_e.set_ylabel("parallel efficiency")
    ax_e.set_title("Parallel efficiency")
    ax_e.set_ylim(0, 1.1)
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=8)

    fig.suptitle(
        "MPI strong scaling: sin(x)·exp(y) over a 1000×1000 OTI grid "
        "(compute region)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
