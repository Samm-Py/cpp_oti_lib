"""Two-dimensional visualization examples for the Python otinum wrapper.

Generated figures are saved as PDFs under:

    python_examples/figures/two_dimensional/
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


def scalar_function(x, y):
    return np.sin(x * y) + 0.35 * x**2 - 0.2 * y**3 + np.exp(-0.25 * (x**2 + y**2))


def oti_function(x, y):
    return oti.sin(x * y) + 0.35 * x**2 - 0.2 * y**3 + oti.exp(-0.25 * (x**2 + y**2))


def evaluate_grid(xs, ys):
    values = np.empty_like(xs)
    dfdx = np.empty_like(xs)
    dfdy = np.empty_like(xs)
    hessian_det = np.empty_like(xs)

    for index in np.ndindex(xs.shape):
        x = T.variable(0, float(xs[index]))
        y = T.variable(1, float(ys[index]))
        f = oti_function(x, y)

        f_xx = f.partial([2, 0])
        f_xy = f.partial([1, 1])
        f_yy = f.partial([0, 2])

        values[index] = f.real()
        dfdx[index] = f.partial([1, 0])
        dfdy[index] = f.partial([0, 1])
        hessian_det[index] = f_xx * f_yy - f_xy**2

    return values, dfdx, dfdy, hessian_det


def taylor_approximation(center, xs, ys):
    x0, y0 = center
    x = T.variable(0, float(x0))
    y = T.variable(1, float(y0))
    f = oti_function(x, y)

    dx = xs - x0
    dy = ys - y0

    return (
        f.real()
        + f.partial([1, 0]) * dx
        + f.partial([0, 1]) * dy
        + 0.5 * f.partial([2, 0]) * dx**2
        + f.partial([1, 1]) * dx * dy
        + 0.5 * f.partial([0, 2]) * dy**2
        + (1.0 / 6.0) * f.partial([3, 0]) * dx**3
        + 0.5 * f.partial([2, 1]) * dx**2 * dy
        + 0.5 * f.partial([1, 2]) * dx * dy**2
        + (1.0 / 6.0) * f.partial([0, 3]) * dy**3
    )


def save_surface_and_contour(xs, ys, values):
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(xs, ys, values, cmap="viridis", linewidth=0.0, antialiased=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$f(x,y)$")
    ax.set_title("Two-dimensional OTI function evaluation")
    fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.12, label=r"$f(x,y)$")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "surface.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    contour = ax.contourf(xs, ys, values, levels=28, cmap="viridis")
    lines = ax.contour(xs, ys, values, levels=10, colors="black", linewidths=0.45, alpha=0.55)
    ax.clabel(lines, inline=True, fontsize=8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Function contours")
    fig.colorbar(contour, ax=ax, label=r"$f(x,y)$")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "contours.pdf")
    plt.close(fig)


def save_gradient_and_hessian(xs, ys, dfdx, dfdy, hessian_det):
    stride = 5
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    magnitude = np.sqrt(dfdx**2 + dfdy**2)
    field = ax.contourf(xs, ys, magnitude, levels=28, cmap="magma")
    ax.quiver(
        xs[::stride, ::stride],
        ys[::stride, ::stride],
        dfdx[::stride, ::stride],
        dfdy[::stride, ::stride],
        color="white",
        scale=45,
        width=0.003,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("OTI gradient field")
    fig.colorbar(field, ax=ax, label=r"$\|\nabla f\|$")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gradient_field.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    limit = np.nanmax(np.abs(hessian_det))
    contour = ax.contourf(
        xs,
        ys,
        hessian_det,
        levels=28,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.contour(xs, ys, hessian_det, levels=[0.0], colors="black", linewidths=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Hessian determinant from OTI derivatives")
    fig.colorbar(contour, ax=ax, label=r"$\det(H_f)$")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "hessian_determinant.pdf")
    plt.close(fig)


def save_taylor_error(xs, ys):
    center = (0.6, -0.35)
    true_values = scalar_function(xs, ys)
    approx_values = taylor_approximation(center, xs, ys)
    error = approx_values - true_values

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    limit = np.nanmax(np.abs(error))
    contour = ax.contourf(xs, ys, error, levels=28, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.plot(center[0], center[1], marker="o", color="black", markersize=4.5, label="expansion point")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Third-order Taylor approximation error")
    fig.colorbar(contour, ax=ax, label="approximation error")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        frameon=False,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(FIGURE_DIR / "third_order_taylor_error.pdf")
    plt.close(fig)


def main():
    grid = np.linspace(-2.0, 2.0, 81)
    xs, ys = np.meshgrid(grid, grid)
    values, dfdx, dfdy, hessian_det = evaluate_grid(xs, ys)

    save_surface_and_contour(xs, ys, values)
    save_gradient_and_hessian(xs, ys, dfdx, dfdy, hessian_det)
    save_taylor_error(xs, ys)

    print(f"saved PDF figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
