#!/usr/bin/env python3
"""Box plot of OTI derivative error vs the analytical derivatives.

Reads the two CSVs emitted by verify_derivatives:
  <prefix>_errors.csv : algebra,abserr   (sampled |OTI - analytical|)
  <prefix>_rmse.csv   : algebra,rmse     (exact RMSE over all points)
and draws one box per algebra (log-scale), annotated with the exact RMSE. The
point: OTI derivatives are exact up to floating-point roundoff, so each box sits
at the algebra's precision floor (double ~1e-16, float ~1e-7).

Usage: plot_accuracy.py <errors.csv> <rmse.csv> <out.png>
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# doubles first, then floats, so the two precision clusters read left-to-right
ORDER = ["<2,1,double>", "<2,2,double>", "<2,1,float>", "<2,2,float>"]


def main(errors_path, rmse_path, out_path):
    errs = defaultdict(list)
    for row in csv.DictReader(open(errors_path)):
        errs[row["algebra"]].append(float(row["abserr"]))
    rmse = {row["algebra"]: float(row["rmse"]) for row in csv.DictReader(open(rmse_path))}

    algebras = [a for a in ORDER if a in errs] + [a for a in errs if a not in ORDER]
    # clamp exact zeros so they are representable on a log axis
    data = [[max(e, 1e-20) for e in errs[a]] for a in algebras]
    colors = ["#1f77b4" if "double" in a else "#d62728" for a in algebras]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bp = ax.boxplot(
        data, tick_labels=algebras, showfliers=True, patch_artist=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        medianprops=dict(color="black"),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    ax.set_yscale("log")
    ax.set_ylabel("|OTI derivative − analytical|")
    ax.set_title("OTI derivative accuracy vs analytical  (f = sin(x)·exp(y))")
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # annotate each box with its exact RMSE
    ymax = max(max(d) for d in data)
    for i, a in enumerate(algebras, start=1):
        ax.text(i, ymax * 3, f"RMSE\n{rmse[a]:.1e}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(top=ymax * 30)

    # precision-floor reference lines
    import numpy as np
    ax.axhline(np.finfo(np.float64).eps, color="#1f77b4", ls=":", lw=1,
               label="float64 eps ≈ 2.2e-16")
    ax.axhline(np.finfo(np.float32).eps, color="#d62728", ls=":", lw=1,
               label="float32 eps ≈ 1.2e-7")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
