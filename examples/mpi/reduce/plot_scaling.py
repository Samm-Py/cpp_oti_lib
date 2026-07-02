#!/usr/bin/env python3
"""Plot compute scaling and Allreduce latency for the reduction tutorial."""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(csv_path="reduce_scaling.csv", out_path="mpi_reduce_scaling.png"):
    rows = read_rows(csv_path)
    modes = {
        "double": ("double", "#4C72B0", "o"),
        "oti_2_2_double": (r"OTI $\langle2,2\rangle$ double", "#DD8452", "s"),
    }

    fig, (ax_speed, ax_latency) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for mode, (label, color, marker) in modes.items():
        selected = sorted(
            (r for r in rows if r["mode"] == mode),
            key=lambda r: int(r["ranks"]),
        )
        ranks = [int(r["ranks"]) for r in selected]
        compute = [float(r["compute_s"]) for r in selected]
        latency = [float(r["allreduce_us"]) for r in selected]
        baseline = compute[0]
        speedup = [baseline / t for t in compute]
        ax_speed.plot(ranks, speedup, marker=marker, color=color, label=label)
        ax_latency.plot(ranks, latency, marker=marker, color=color, label=label)

    ranks = sorted({int(r["ranks"]) for r in rows})
    ax_speed.plot(ranks, ranks, "--", color="0.55", label="ideal")
    ax_speed.set_xlabel("MPI ranks")
    ax_speed.set_ylabel("local-compute speedup")
    ax_speed.set_xscale("log", base=2)
    ax_speed.set_xticks(ranks, labels=[str(r) for r in ranks])
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(frameon=False)

    ax_latency.set_xlabel("MPI ranks")
    ax_latency.set_ylabel(r"one-element Allreduce latency [$\mu$s]")
    ax_latency.set_xscale("log", base=2)
    ax_latency.set_yscale("log")
    ax_latency.set_xticks(ranks, labels=[str(r) for r in ranks])
    ax_latency.grid(True, alpha=0.25)
    ax_latency.legend(frameon=False)

    fig.suptitle(
        "Global QoI reduction: local work scales; one-jet collective stays small"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
