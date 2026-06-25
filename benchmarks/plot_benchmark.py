#!/usr/bin/env python3
"""Plot one otinum isolation-benchmark CSV: metric value vs algebra size, one
line per variant, faceted by kernel (rows) x precision (columns). Works for any
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
    out = []
    for r in rows:
        # Tolerate CSVs concatenated from several binaries (e.g. the arithmetic
        # paths or the two alignment rules): each binary prints its own header,
        # so skip the repeated header lines that land mid-file.
        if r["backend"] == "backend":
            continue
        for k in ("M", "N", "ncoeffs", "nproducts", "repetition"):
            r[k] = int(r[k])
        r["value"] = float(r["value"])
        out.append(r)
    return out


# Byte size of each coefficient type, for the alignment-promotion test below.
COEFF_SIZEOF = {"float": 4, "double": 8, "long double": 16}


def is_promoted(ncoeffs, coeff_type):
    """True if otinum_alignment promotes this shape above natural alignment,
    i.e. the conditional alignas rule changes the layout (natural != aligned).
    Mirrors detail::otinum_alignment in include/otinum/core.hpp."""
    sz = COEFF_SIZEOF.get(coeff_type)
    if sz is None:
        return False
    nbytes = sz * ncoeffs
    aligned = (
        32 if nbytes % 32 == 0
        else 16 if nbytes % 16 == 0
        else max(8, sz) if nbytes % 8 == 0
        else sz
    )
    return aligned != sz  # natural alignment is alignof(Coeff) == sz


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

    # Only an alignment plot (natural vs aligned) has a meaningful notion of
    # "this ncoeffs gets promoted"; shade those x columns so the unchanged
    # control shapes are visually distinct.
    mark_promotion = (xkey == "ncoeffs" and "natural" in variants
                      and "aligned" in variants)

    # Kernels as rows, precision as columns: with many kernels this keeps the
    # figure tall rather than wide, so each panel stays readable when the image
    # is scaled to page width in the docs.
    nrow, ncol = len(kernels), len(precisions)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.6 * nrow),
                             squeeze=False, sharex=True)
    for i, kern in enumerate(kernels):
        for j, prec in enumerate(precisions):
            ax = axes[i][j]
            if mark_promotion:
                promoted = sorted({r[xkey] for r in rows
                                   if r["coeff_type"] == prec
                                   and is_promoted(r["ncoeffs"], prec)})
                for n in promoted:
                    ax.axvspan(n / 1.08, n * 1.08, color="0.5", alpha=0.15,
                               zorder=0)
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
            ax.set_title(f"{kern}, {prec}")
            if i == nrow - 1:
                ax.set_xlabel(xkey)
            if j == 0:
                ax.set_ylabel(metric + ("  (lower better)" if lower_better else "  (higher better)"))
    handles, _ = axes[0][0].get_legend_handles_labels()
    if mark_promotion:
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(color="0.5", alpha=0.3,
                                      label="promoted (natural != aligned)"))
    axes[0][0].legend(handles=handles, fontsize=9, loc="best")
    fig.suptitle(path.stem)
    # Reserve a sliver at the top so the suptitle never overlaps the first-row
    # axes titles, which tight_layout does not account for on its own.
    fig.tight_layout(rect=(0, 0, 1, 0.99))
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
