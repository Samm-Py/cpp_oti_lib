#!/usr/bin/env python3
"""Plot the analytic validity intuition-builders. Usage: python plot_validity_examples.py
Reads data/*.csv (written by ./validity_examples) -> figures/ex{1..5}.png"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.makedirs("figures", exist_ok=True)


def col(rows, k, t=float):
    return np.array([t(r[k]) for r in rows])


def grid(name):
    rows = list(csv.DictReader(open(f"data/{name}")))
    hx = sorted(set(float(r["hx"]) for r in rows))
    hy = sorted(set(float(r["hy"]) for r in rows))
    nx, ny = len(hx), len(hy)
    X = np.array(hx)
    Y = np.array(hy)
    R = np.zeros((ny, nx))
    T = np.zeros((ny, nx))
    ix = {v: i for i, v in enumerate(hx)}
    iy = {v: i for i, v in enumerate(hy)}
    for r in rows:
        R[iy[float(r["hy"])], ix[float(r["hx"])]] = float(r["real_err"])
        T[iy[float(r["hy"])], ix[float(r["hx"])]] = float(r["trusted"])
    return X, Y, R, T


params = {r["name"]: r for r in csv.DictReader(open("data/params.csv"))}

# ---- Example 1: 1D exp(x) ----
rows = list(csv.DictReader(open("data/ex1.csv")))
h = col(rows, "h"); truth = col(rows, "truth"); lin = col(rows, "linear")
real = col(rows, "real_err"); est = col(rows, "est_err"); reach = float(rows[0]["reach"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(h, truth, "k", label="truth $e^h$")
ax[0].plot(h, lin, "C0--", label="linear surrogate $1+h$")
ax[0].axvspan(-reach, reach, color="C2", alpha=0.15, label=f"trusted |h|<{reach:.3f}")
ax[0].legend(); ax[0].set_xlabel("step h"); ax[0].set_title("exp(x) around 0: surrogate & trusted interval")
ax[1].plot(h, np.abs(real), "k", label="|real error|")
ax[1].plot(h, np.abs(est), "C1--", label="|estimated| $=\\frac{1}{2} h^2$")
ax[1].axhline(0.05 * 1.0, color="C3", ls=":", label="budget $\\tau|f|$")
ax[1].axvline(reach, color="C2"); ax[1].axvline(-reach, color="C2")
ax[1].legend(); ax[1].set_xlabel("step h"); ax[1].set_yscale("log")
ax[1].set_title("error vs budget; reach = where they cross")
fig.tight_layout(); fig.savefig("figures/ex1.png", dpi=120); plt.close(fig)

# ---- Example 2: reach vs curvature & vs singularity ----
ca = list(csv.DictReader(open("data/ex2_curvature.csv")))
sg = list(csv.DictReader(open("data/ex2_singularity.csv")))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].loglog(col(ca, "k"), col(ca, "reach"), "o-", label="reach")
ax[0].loglog(col(ca, "k"), 0.3162 / col(ca, "k"), "k:", label="$\\sqrt{2\\tau}/k$")
ax[0].set_xlabel("curvature k  (exp(kx))"); ax[0].set_ylabel("reach")
ax[0].legend(); ax[0].set_title("reach $\\propto 1/k$: sharper curvature, smaller reach")
ax[1].plot(col(sg, "distance_to_sing"), col(sg, "reach"), "o-")
ax[1].set_xlabel("distance to pole (1 - $x_0$)"); ax[1].set_ylabel("reach")
ax[1].set_title("1/(1-x): reach collapses approaching the singularity")
ax[1].invert_xaxis()
fig.tight_layout(); fig.savefig("figures/ex2.png", dpi=120); plt.close(fig)


def plot_2d(name, title, draw_box=False, note=None):
    X, Y, R, T = grid(f"{name}.csv")
    p = params.get(name)
    tau, f0 = float(p["tau"]), float(p["f0"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cf = ax.contourf(X, Y, np.abs(R), levels=25, cmap="magma")
    plt.colorbar(cf, ax=ax, label="|real error|")
    # estimated trusted region boundary (is_trusted) vs the real budget level set
    ax.contour(X, Y, T, levels=[0.5], colors="cyan", linewidths=2)
    ax.contour(X, Y, np.abs(R), levels=[tau * abs(f0)], colors="lime",
               linewidths=1.5, linestyles="--")
    if draw_box:
        rx, ry = float(p["rx"]), float(p["ry"])
        if np.isfinite(rx) and np.isfinite(ry):
            ax.add_patch(Rectangle((-rx, -ry), 2 * rx, 2 * ry, fill=False, ec="white",
                                   ls=":", lw=1.5))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="cyan", lw=2, label="estimated trust (is_trusted)"),
               Line2D([], [], color="lime", ls="--", label="real budget level set")]
    if draw_box:
        handles.append(Line2D([], [], color="white", ls=":", label="per-axis box (corners overshoot)"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xlabel("$h_x$"); ax.set_ylabel("$h_y$"); ax.set_title(title)
    ax.set_aspect("equal")
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color="white",
                va="bottom")
    fig.tight_layout(); fig.savefig(f"figures/{name}.png", dpi=120); plt.close(fig)


# ---- Example 3: coupling -- tilted region, box corners overshoot ----
plot_2d("ex3", "exp(x)+exp(y)+0.5xy: estimated vs real trust (box corners overshoot)",
        draw_box=True)
# ---- Example 4: x is free (infinite reach), y is the constraint ----
plot_2d("ex4", "5+3x+y$^2$: error independent of $h_x$ (x free), bounded band in $h_y$")
# ---- Example 5: pure quadratic -- estimate exact, trust = exact ellipse ----
plot_2d("ex5", "1+2x-y+x$^2$+xy+y$^2$: order-2 estimate EXACT (cyan == green)",
        note="estimated and real boundaries coincide (residual ~ 1e-16)")

print("wrote figures/ex1.png .. ex5.png")
