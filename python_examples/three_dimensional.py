"""Three-dimensional visualization examples for the Python otinum wrapper.

Generated figures are saved as PDFs under:

    python_examples/figures/three_dimensional/
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

T = oti.OTI_3_2


def scalar_function(x, y, z):
    radius_sq = x**2 + 0.7 * y**2 + 1.2 * z**2
    return np.sin(x * y) + 0.45 * np.cos(y * z) + 0.2 * x * z + np.exp(-0.35 * radius_sq)


def oti_function(x, y, z):
    radius_sq = x**2 + 0.7 * y**2 + 1.2 * z**2
    return oti.sin(x * y) + 0.45 * oti.cos(y * z) + 0.2 * x * z + oti.exp(-0.35 * radius_sq)


def evaluate_slice(xs, ys, z0):
    values = np.empty_like(xs)
    grad_norm = np.empty_like(xs)
    laplacian = np.empty_like(xs)

    for index in np.ndindex(xs.shape):
        x = T.variable(0, float(xs[index]))
        y = T.variable(1, float(ys[index]))
        z = T.variable(2, float(z0))
        f = oti_function(x, y, z)

        fx = f.partial([1, 0, 0])
        fy = f.partial([0, 1, 0])
        fz = f.partial([0, 0, 1])

        values[index] = f.real()
        grad_norm[index] = np.sqrt(fx**2 + fy**2 + fz**2)
        laplacian[index] = (
            f.partial([2, 0, 0])
            + f.partial([0, 2, 0])
            + f.partial([0, 0, 2])
        )

    return values, grad_norm, laplacian


def evaluate_volume(points):
    values = np.empty(points.shape[0])
    grad_norm = np.empty(points.shape[0])

    for i, (xv, yv, zv) in enumerate(points):
        x = T.variable(0, float(xv))
        y = T.variable(1, float(yv))
        z = T.variable(2, float(zv))
        f = oti_function(x, y, z)

        fx = f.partial([1, 0, 0])
        fy = f.partial([0, 1, 0])
        fz = f.partial([0, 0, 1])

        values[i] = f.real()
        grad_norm[i] = np.sqrt(fx**2 + fy**2 + fz**2)

    return values, grad_norm


def taylor_slice(center, xs, ys):
    x0, y0, z0 = center
    x = T.variable(0, float(x0))
    y = T.variable(1, float(y0))
    z = T.variable(2, float(z0))
    f = oti_function(x, y, z)

    dx = xs - x0
    dy = ys - y0
    dz = 0.0

    return (
        f.real()
        + f.partial([1, 0, 0]) * dx
        + f.partial([0, 1, 0]) * dy
        + f.partial([0, 0, 1]) * dz
        + 0.5 * f.partial([2, 0, 0]) * dx**2
        + f.partial([1, 1, 0]) * dx * dy
        + f.partial([1, 0, 1]) * dx * dz
        + 0.5 * f.partial([0, 2, 0]) * dy**2
        + f.partial([0, 1, 1]) * dy * dz
        + 0.5 * f.partial([0, 0, 2]) * dz**2
    )


def save_slice_fields(xs, ys, z0, values, grad_norm, laplacian):
    fields = [
        (values, r"$f(x,y,z_0)$", "viridis", "field_slice.pdf"),
        (grad_norm, r"$\|\nabla f\|$", "magma", "gradient_norm_slice.pdf"),
        (laplacian, r"$\nabla^2 f$", "coolwarm", "laplacian_slice.pdf"),
    ]

    for field, label, cmap, filename in fields:
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        kwargs = {}
        if cmap == "coolwarm":
            limit = np.nanmax(np.abs(field))
            kwargs = {"vmin": -limit, "vmax": limit}
        contour = ax.contourf(xs, ys, field, levels=28, cmap=cmap, **kwargs)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(rf"Three-dimensional scalar field slice, $z_0={z0}$")
        fig.colorbar(contour, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename)
        plt.close(fig)


def save_threshold_scatter():
    grid = np.linspace(-1.6, 1.6, 25)
    xg, yg, zg = np.meshgrid(grid, grid, grid, indexing="ij")
    points = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    values, grad_norm = evaluate_volume(points)
    threshold = np.quantile(values, 0.82)
    selected = values >= threshold

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[selected, 0],
        points[selected, 1],
        points[selected, 2],
        c=grad_norm[selected],
        cmap="plasma",
        s=13,
        alpha=0.88,
        linewidths=0.0,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title("High-value field region colored by gradient norm")
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.12, label=r"$\|\nabla f\|$")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "high_value_region.pdf")
    plt.close(fig)


def save_taylor_error_slice(xs, ys):
    center = (0.35, -0.45, 0.4)
    true_values = scalar_function(xs, ys, center[2])
    approx_values = taylor_slice(center, xs, ys)
    error = approx_values - true_values

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    limit = np.nanmax(np.abs(error))
    contour = ax.contourf(xs, ys, error, levels=28, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.plot(center[0], center[1], marker="o", color="black", markersize=4.5, label="expansion point")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Second-order Taylor error on $z=z_0$ slice")
    fig.colorbar(contour, ax=ax, label="approximation error")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        frameon=False,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(FIGURE_DIR / "second_order_taylor_error_slice.pdf")
    plt.close(fig)


def main():
    grid = np.linspace(-1.6, 1.6, 75)
    xs, ys = np.meshgrid(grid, grid)
    z0 = 0.4
    values, grad_norm, laplacian = evaluate_slice(xs, ys, z0)

    save_slice_fields(xs, ys, z0, values, grad_norm, laplacian)
    save_threshold_scatter()
    save_taylor_error_slice(xs, ys)

    print(f"saved PDF figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
