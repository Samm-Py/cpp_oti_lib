"""One-dimensional visualization examples for the Python otinum wrapper.

Generated figures are saved as PDFs under:

    python_examples/figures/one_dimensional/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import otinum as oti


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

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figures" / Path(__file__).stem
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem):
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300)


def scalar_function(x):
    return np.sin(1.3 * x) + 0.25 * x**3 - 0.8 * np.exp(-0.5 * x)


def oti_function(x):
    return oti.sin(1.3 * x) + 0.25 * x**3 - 0.8 * oti.exp(-0.5 * x)


def evaluate_with_derivatives(points):
    values = []
    first = []
    second = []
    third = []

    for point in points:
        x = oti.OTI_1_3.variable(0, float(point))
        f = oti_function(x)
        values.append(f.real())
        first.append(f.partial([1]))
        second.append(f.partial([2]))
        third.append(f.partial([3]))

    return (
        np.asarray(values),
        np.asarray(first),
        np.asarray(second),
        np.asarray(third),
    )


def taylor_approximation(center, offsets):
    x = oti.OTI_1_3.variable(0, float(center))
    f = oti_function(x)

    return (
        f.real()
        + f.partial([1]) * offsets
        + 0.5 * f.partial([2]) * offsets**2
        + (1.0 / 6.0) * f.partial([3]) * offsets**3
    )


def save_function_and_derivatives():
    xs = np.linspace(-3.0, 3.0, 500)
    values, first, second, third = evaluate_with_derivatives(xs)
    marker_xs = np.linspace(-3.0, 3.0, 19)
    marker_values = evaluate_with_derivatives(marker_xs)

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.8), sharex=True)
    series = [
        (values, marker_values[0], r"$f(x)$"),
        (first, marker_values[1], r"$f'(x)$"),
        (second, marker_values[2], r"$f''(x)$"),
        (third, marker_values[3], r"$f'''(x)$"),
    ]

    for ax, (ys, marker_ys, label) in zip(axes, series):
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8)
        ax.plot(
            marker_xs,
            marker_ys,
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor="#d62728",
            markeredgewidth=1.0,
        )
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(r"$x$")
    line_handle = plt.Line2D([], [], color="#1f77b4", linewidth=1.8, label="continuous evaluation")
    marker_handle = plt.Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor="#d62728",
        markeredgewidth=1.0,
        label="OTI samples",
    )
    fig.legend(
        handles=[line_handle, marker_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        handlelength=2.4,
    )
    fig.suptitle("OTI-derived function values and derivatives")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.98), h_pad=0.8)
    save_figure(fig, "function_and_derivatives")
    plt.close(fig)


def save_taylor_comparison():
    center = 0.75
    xs = np.linspace(center - 1.75, center + 1.75, 500)
    offsets = xs - center

    true_values = scalar_function(xs)
    approx_values = taylor_approximation(center, offsets)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, true_values, label="true function", color="#1f77b4", linewidth=2.0)
    ax.plot(
        xs,
        approx_values,
        label="third-order Taylor approximation",
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
    )
    ax.axvline(center, color="#444444", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("value")
    ax.set_title("Local Taylor approximation from OTI coefficients")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=2.8,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, "third_order_taylor_comparison")
    plt.close(fig)


def save_taylor_error():
    center = 0.75
    xs = np.linspace(center - 1.75, center + 1.75, 500)
    offsets = xs - center

    error = taylor_approximation(center, offsets) - scalar_function(xs)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, error, color="#2ca02c", linewidth=2.0, label="approximation error")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.axvline(center, color="#444444", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Taylor approximation error")
    ax.set_title("Error of third-order OTI Taylor approximation")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        handlelength=2.8,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, "third_order_taylor_error")
    plt.close(fig)


def main():
    save_function_and_derivatives()
    save_taylor_comparison()
    save_taylor_error()
    print(f"saved PDF figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
