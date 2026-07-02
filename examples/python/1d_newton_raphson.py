"""Newton-Raphson example with OTI design-parameter sensitivities.

The scalar state u solves:

    F(u, a, b) = u^3 + a u - b = 0

The design parameters a and b are represented by an OTI_2_3 value, so the
converged root contains derivatives with respect to both parameters through
third total order.

Generated figures are saved as PDFs under:

    python_examples/figures/1d_newton_raphson/
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

T = oti.OTI_2_3


def residual(u, a, b):
    return u**3.0 + a * u - b


def residual_du(u, a):
    return 3.0 * u**2.0 + a


def real_root(a, b, initial=1.0, tolerance=1e-14, max_iter=30):
    u = float(initial)
    for _ in range(max_iter):
        f = u**3 + a * u - b
        df = 3.0 * u**2 + a
        step = f / df
        u -= step
        if abs(step) < tolerance:
            break
    return u


def derivative_snapshot(u):
    return {
        r"$\partial u / \partial a$": u.partial([1, 0]),
        r"$\partial u / \partial b$": u.partial([0, 1]),
        r"$\partial^2 u / \partial a^2$": u.partial([2, 0]),
        r"$\partial^2 u / \partial a \partial b$": u.partial([1, 1]),
        r"$\partial^2 u / \partial b^2$": u.partial([0, 2]),
    }


def solve_with_oti_newton(
    a0=1.4,
    b0=2.0,
    initial=0.5,
    residual_tolerance=1e-12,
    derivative_tolerance=1e-12,
    max_iter=14,
):
    a = T.variable(0, a0)
    b = T.variable(1, b0)
    u = T(initial)

    history = []
    previous_derivatives = np.asarray(u.data()[1:])
    stop_iteration = None

    for iteration in range(max_iter + 1):
        f = residual(u, a, b)
        derivative_coeffs = np.asarray(u.data()[1:])
        derivative_delta = np.max(np.abs(derivative_coeffs - previous_derivatives))
        snapshot = derivative_snapshot(u)

        history.append(
            {
                "iteration": iteration,
                "u": u,
                "real": u.real(),
                "residual": abs(f.real()),
                "step": np.nan,
                "derivative_delta": derivative_delta,
                "derivatives": snapshot,
            }
        )

        if (
            iteration > 0
            and stop_iteration is None
            and abs(f.real()) < residual_tolerance
            and derivative_delta < derivative_tolerance
        ):
            stop_iteration = iteration

        if iteration == max_iter:
            break

        previous_derivatives = derivative_coeffs
        step = f / residual_du(u, a)
        u = u - step
        history[-1]["step"] = abs(step.real())

    return history, stop_iteration


def save_convergence_plot(history, stop_iteration):
    iterations = np.asarray([row["iteration"] for row in history])
    residuals = np.asarray([row["residual"] for row in history])
    derivative_delta = np.asarray([row["derivative_delta"] for row in history])
    steps = np.asarray([row["step"] for row in history])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.semilogy(iterations, residuals, marker="o", markersize=4.0, label=r"$|F(u_k)|$")
    ax.semilogy(
        iterations,
        np.where(np.isfinite(steps), steps, np.nan),
        marker="s",
        markersize=4.0,
        label=r"$|\Delta u_k|$",
    )
    ax.semilogy(
        iterations,
        np.maximum(derivative_delta, np.finfo(float).tiny),
        marker="^",
        markersize=4.0,
        label="max derivative-coefficient change",
    )

    if stop_iteration is not None:
        ax.axvline(stop_iteration, color="#444444", linewidth=1.0, linestyle=":")
        ax.text(
            stop_iteration + 0.1,
            2e-13,
            f"stop: {stop_iteration}",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=10,
        )

    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("convergence metric")
    ax.set_title("Newton convergence of state and derivative coefficients")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(FIGURE_DIR / "newton_convergence_metrics.pdf")
    plt.close(fig)


def save_derivative_history_plot(history, stop_iteration):
    iterations = np.asarray([row["iteration"] for row in history])
    derivative_names = list(history[-1]["derivatives"].keys())

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name in derivative_names:
        values = [row["derivatives"][name] for row in history]
        ax.plot(iterations, values, marker="o", markersize=3.5, linewidth=1.6, label=name)

    if stop_iteration is not None:
        ax.axvline(stop_iteration, color="#444444", linewidth=1.0, linestyle=":")

    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("derivative value")
    ax.set_title("Convergence of selected design sensitivities")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(FIGURE_DIR / "derivative_history.pdf")
    plt.close(fig)


def save_taylor_design_slice(history):
    root = history[-1]["u"]
    a0 = 1.4
    b0 = 2.0
    offsets = np.linspace(-0.55, 0.55, 300)
    a_values = a0 + offsets
    true_roots = np.asarray([real_root(a, b0, initial=root.real()) for a in a_values])

    taylor_roots = (
        root.real()
        + root.partial([1, 0]) * offsets
        + 0.5 * root.partial([2, 0]) * offsets**2
        + (1.0 / 6.0) * root.partial([3, 0]) * offsets**3
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(a_values, true_roots, color="#1f77b4", linewidth=2.0, label="Newton root")
    ax.plot(
        a_values,
        taylor_roots,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="third-order OTI Taylor slice",
    )
    ax.axvline(a0, color="#444444", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"design parameter $a$")
    ax.set_ylabel(r"root $u(a, b_0)$")
    ax.set_title("Root variation with one design parameter")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=2.8,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(FIGURE_DIR / "design_parameter_taylor_slice.pdf")
    plt.close(fig)


def main():
    history, stop_iteration = solve_with_oti_newton()
    save_convergence_plot(history, stop_iteration)
    save_derivative_history_plot(history, stop_iteration)
    save_taylor_design_slice(history)

    final = history[-1]
    print(f"saved PDF figures to {FIGURE_DIR}")
    print(f"selected stopping iteration: {stop_iteration}")
    print(f"root: {final['real']:.16g}")
    print(f"residual: {final['residual']:.3e}")
    for name, value in final["derivatives"].items():
        print(f"{name}: {value:.16g}")


if __name__ == "__main__":
    main()
