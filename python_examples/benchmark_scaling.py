from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "benchmarks" / "results" / "elementary_scaling.csv"
FIGURE_DIR = ROOT / "python_examples" / "figures" / "benchmark_scaling"


CASE_LABELS = {
    "first_order_simple": r"First order: polynomial",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rows_for_case(rows: list[dict[str, str]], case_name: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["case"] == case_name]
    return sorted(selected, key=lambda row: int(row["dimension"]))


def plot_case(rows: list[dict[str, str]], case_name: str) -> None:
    selected = rows_for_case(rows, case_name)
    dims = [int(row["dimension"]) for row in selected]
    ratios = [float(row["ratio"]) for row in selected]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.bar(np.arange(len(dims)), ratios, color="#3b6ea8", width=0.68)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([str(dim) for dim in dims])
    ax.set_xlabel(r"Number of variables, $m$")
    ax.set_ylabel(r"Runtime ratio, OTI / real")
    ax.set_title(CASE_LABELS[case_name])
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(FIGURE_DIR / f"{case_name}_runtime_ratio.pdf")
    plt.close(fig)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {DATA_FILE}. Run benchmarks/elementary_scaling.cpp first."
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = load_rows(DATA_FILE)

    for case_name in CASE_LABELS:
        plot_case(rows, case_name)

    print(f"saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
