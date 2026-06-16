#!/usr/bin/env python3
"""Plot one otinum isolation-benchmark CSV: metric value vs algebra size, one
line per variant, faceted by precision (rows) x kernel (columns). Works for any
of the suite's CSVs since they share the schema.

  python3 plot_benchmark.py results/bench_layout.csv
  python3 plot_benchmark.py results/            # plots every bench_*.csv
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="a bench_*.csv or a directory of them")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--x", choices=["ncoeffs", "M", "nproducts"], default="ncoeffs")
    return p.parse_args()


def load(path):
    with path.open(newline="") as h:
        rows = list(csv.DictReader(h))
    for r in rows:
        for k in ("M", "N", "ncoeffs", "nproducts", "repetition"):
            r[k] = int(r[k])
        r["value"] = float(r["value"])
    return rows


def plot_one(path, output_dir, xkey):
    rows = load(path)
    if not rows:
        print(f"{path.name}: empty, skipping")
        return
    metric = rows[0]["metric"]
    lower_better = metric.startswith("ns")

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precisions = sorted({r["coeff_type"] for r in rows})
    kernels = sorted({r["kernel"] for r in rows})
    variants = sorted({r["variant"] for r in rows})

    # (precision, kernel, variant) -> sorted [(x, median value)]
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["coeff_type"], r["kernel"], r["variant"])][r[xkey]].append(r["value"])
    series = {k: sorted((x, statistics.median(v)) for x, v in byx.items())
              for k, byx in agg.items()}

    nrow, ncol = len(precisions), len(kernels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.2 * nrow),
                             squeeze=False, sharex=True)
    for i, prec in enumerate(precisions):
        for j, kern in enumerate(kernels):
            ax = axes[i][j]
            for variant in variants:
                s = series.get((prec, kern, variant))
                if not s:
                    continue
                xs = [p[0] for p in s]
                ys = [p[1] for p in s]
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6, label=variant)
            ax.set_xscale("log")
            if lower_better:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{prec}, {kern}")
            if i == nrow - 1:
                ax.set_xlabel(xkey)
            if j == 0:
                ax.set_ylabel(metric + ("  (lower better)" if lower_better else "  (higher better)"))
    axes[0][0].legend(fontsize=9, loc="best")
    fig.suptitle(path.stem)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        out = output_dir / f"{path.stem}.{suffix}"
        fig.savefig(out, dpi=200 if suffix == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    args = parse_args()
    inp = args.input
    paths = sorted(inp.glob("bench_*.csv")) if inp.is_dir() else [inp]
    if not paths:
        raise SystemExit(f"no bench_*.csv found in {inp}")
    output_dir = (args.output_dir or (inp if inp.is_dir() else inp.parent)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        plot_one(p, output_dir, args.x)


if __name__ == "__main__":
    main()
