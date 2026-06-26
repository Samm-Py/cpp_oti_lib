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


def csv_rows(name):
    return list(csv.DictReader(open(f"data/{name}")))


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

# ---- Example 1: 1D exp(k x), swept over curvature, fully annotated ----
# The OTI variable is x; the horizontal axis is the STEP h = displacement from the
# expansion point x0=0. Certified model is LINEAR (1+kh); the order-2 term is the
# error estimate. Budget = tau*|f0|; reach r solves (1/2)k^2 r^2 = budget.
def plot_ex1(csv_name, outpng, headline):
    rows = csv_rows(csv_name)
    h = col(rows, "h"); truth = col(rows, "truth"); lin = col(rows, "linear")
    real = col(rows, "real_err"); est = col(rows, "est_err")
    reach = float(rows[0]["reach"]); k = float(rows[0]["k"]); budget = float(rows[0]["budget"])
    ks = "" if k == 1 else f"{k:g}"  # drop the redundant coefficient when k==1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    # -- left: certified linear surrogate vs truth, trusted interval --
    ax[0].plot(h, truth, "k", lw=2, label=f"truth $e^{{{ks}h}}$")
    ax[0].plot(h, lin, "C0--", lw=2, label=f"linear surrogate $1+{ks}h$")
    ax[0].axvspan(-reach, reach, color="C2", alpha=0.15,
                  label=f"trusted  $|h|<r={reach:.3f}$")
    ax[0].axvline(0, color="0.6", lw=0.8)
    ax[0].annotate("expansion point $x_0=0$\n(OTI variable is $x$)", xy=(0, lin[len(lin) // 2]),
                   xytext=(0.55, 0.06), textcoords="axes fraction", fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="0.4"))
    ax[0].set_xlabel("step $h$  (displacement from $x_0$ — NOT the OTI variable)")
    ax[0].set_ylabel("value")
    ax[0].set_title("certified LINEAR surrogate (model_order=1) & trusted interval")

    # -- right: error vs budget, reach = where estimate meets budget --
    ax[1].plot(h, np.abs(real), "k", lw=2, label="$|$real error$|=|e^{kh}-(1+kh)|$")
    ax[1].plot(h, np.abs(est), "C1--", lw=2,
               label=r"$|$estimated$|=\frac{1}{2}k^2h^2$ (order-2 term)")
    # budget & reach get LABELS (not in-axes annotations) so they land in the
    # bottom-margin legend and never collide with the pinched-in green lines.
    ax[1].axhline(budget, color="C3", ls=":", lw=2,
                  label=rf"budget $=\tau|f_0|={budget:.3g}$")
    ax[1].axvline(reach, color="C2", lw=2,
                  label=rf"reach $r={reach:.3f}=\sqrt{{2\tau}}/k$")
    ax[1].axvline(-reach, color="C2", lw=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step $h$"); ax[1].set_ylabel("absolute error")
    ax[1].set_title("error vs budget — green = reach, where estimate crosses budget")

    # legends live in the bottom margin so they never cover the curves/annotations
    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, fontsize=8,
               frameon=False)
    fig.legend(h1, l1, loc="lower right", bbox_to_anchor=(0.97, 0.0), ncol=2, fontsize=8,
               frameon=False)
    fig.suptitle(headline, fontsize=12)
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    fig.savefig(outpng, dpi=120); plt.close(fig)


plot_ex1("ex1_lo.csv", "figures/ex1_lo.png",
         "Example 1 — LOW curvature $k=0.5$: gentle bend, large trusted step")
plot_ex1("ex1.csv", "figures/ex1.png",
         "Example 1 — baseline $k=1$: $\\exp(x)$ hello-world")
plot_ex1("ex1_hi.csv", "figures/ex1_hi.png",
         "Example 1 — HIGH curvature $k=2$: sharp bend, small trusted step")

# combined overlay: reach shrinks as curvature grows (reach ~ 1/k)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for csv_name, c in (("ex1_lo.csv", "C0"), ("ex1.csv", "C2"), ("ex1_hi.csv", "C3")):
    rows = csv_rows(csv_name)
    h = col(rows, "h"); est = col(rows, "est_err")
    reach = float(rows[0]["reach"]); k = float(rows[0]["k"])
    ax.plot(h, np.abs(est), color=c, lw=2,
            label=f"$k={k:g}$:  $\\frac{{1}}{{2}} k^2h^2$,  reach$={reach:.3f}$")
    ax.axvline(reach, color=c, ls="--", lw=1); ax.axvline(-reach, color=c, ls="--", lw=1)
budget = float(csv_rows("ex1.csv")[0]["budget"])
ax.axhline(budget, color="k", ls=":", lw=2, label=f"budget $\\tau|f_0|={budget:.3g}$")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
ax.set_xlabel("step $h$"); ax.set_ylabel("estimated error  $\\frac{1}{2} k^2h^2$")
ax.set_title("reach $\\propto 1/k$: higher curvature → smaller trusted step")
ax.legend(fontsize=9, loc="lower right"); fig.tight_layout()
fig.savefig("figures/ex1_curvature.png", dpi=120); plt.close(fig)

# ---- Example 2: a nearby pole collapses the reach (1/(1-x) toward x=1) ----
# ex1 format, but the function has a singularity at x=1 and we expand at several x0
# marching toward it. reach r = sqrt(tau)*d (d = 1-x0). The real error is ASYMMETRIC:
# toward the pole (+h) it blows past the symmetric quadratic estimate, so the
# certified reach is optimistic on the pole side, conservative away from it.
def plot_ex2(csv_name, outpng, headline):
    rows = csv_rows(csv_name)
    h = col(rows, "h"); truth = col(rows, "truth"); lin = col(rows, "linear")
    real = col(rows, "real_err"); est = col(rows, "est_err")
    reach = float(rows[0]["reach"]); x0 = float(rows[0]["x0"])
    d = float(rows[0]["d"]); budget = float(rows[0]["budget"]); f0 = float(rows[0]["f0"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    # -- left: truth (with the nearby pole) vs linear surrogate, trusted interval --
    ax[0].plot(h, truth, "k", lw=2, label=r"truth $1/(1-x)$")
    ax[0].plot(h, lin, "C0--", lw=2, label="linear surrogate")
    ax[0].axvspan(-reach, reach, color="C2", alpha=0.15,
                  label=f"trusted  $|h|<r={reach:.3f}$")
    ax[0].axvline(0, color="0.6", lw=0.8)
    ax[0].annotate(f"expansion point $x_0={x0:g}$\n(dist $d={d:g}$ to pole at $x=1$)",
                   xy=(0, f0), xytext=(0.04, 0.72), textcoords="axes fraction", fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="0.4"))
    ax[0].set_xlabel("step $h$  (displacement from $x_0$; pole at $h=d$)")
    ax[0].set_ylabel("value")
    ax[0].set_title(f"linear surrogate vs truth near a pole ($d={d:g}$)")

    # -- right: error vs budget; ASYMMETRIC because the pole is on the +h side --
    ax[1].plot(h, np.abs(real), "k", lw=2, label="$|$real error$|$")
    ax[1].plot(h, np.abs(est), "C1--", lw=2, label=r"$|$estimated$|=|c_2|h^2$ (symmetric)")
    ax[1].axhline(budget, color="C3", ls=":", lw=2,
                  label=rf"budget $=\tau|f_0|={budget:.3g}$")
    ax[1].axvline(reach, color="C2", lw=2,
                  label=rf"reach $r={reach:.3f}=\sqrt{{\tau}}\,d$")
    ax[1].axvline(-reach, color="C2", lw=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step $h$  (pole at $+d$)"); ax[1].set_ylabel("absolute error")
    ax[1].set_title("error grows faster toward the pole ($+h$): reach is optimistic there")

    h0, l0 = ax[0].get_legend_handles_labels()
    h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, fontsize=8,
               frameon=False)
    fig.legend(h1, l1, loc="lower right", bbox_to_anchor=(0.97, 0.0), ncol=2, fontsize=8,
               frameon=False)
    fig.suptitle(headline, fontsize=12)
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    fig.savefig(outpng, dpi=120); plt.close(fig)


plot_ex2("ex2_far.csv", "figures/ex2_far.png",
         "Example 2 — FAR from pole $x_0=0$ ($d=1$): large reach")
plot_ex2("ex2_mid.csv", "figures/ex2_mid.png",
         "Example 2 — approaching $x_0=0.5$ ($d=0.5$)")
plot_ex2("ex2_near.csv", "figures/ex2_near.png",
         "Example 2 — NEAR pole $x_0=0.8$ ($d=0.2$): reach collapses")

# combined overlay: relative error (common budget = tau), reach shrinks with d
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for csv_name, c in (("ex2_far.csv", "C0"), ("ex2_mid.csv", "C2"), ("ex2_near.csv", "C3")):
    rows = csv_rows(csv_name)
    h = col(rows, "h"); est = col(rows, "est_err"); f0 = float(rows[0]["f0"])
    reach = float(rows[0]["reach"]); d = float(rows[0]["d"])
    ax.plot(h, np.abs(est) / f0, color=c, lw=2, label=f"$d={d:g}$:  reach$={reach:.3f}$")
    ax.axvline(reach, color=c, ls="--", lw=1); ax.axvline(-reach, color=c, ls="--", lw=1)
ax.axhline(0.05, color="k", ls=":", lw=2, label=r"budget $\tau=0.05$ (relative)")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
ax.set_xlabel("step $h$"); ax.set_ylabel(r"relative estimated error  $|c_2|h^2/|f_0|$")
ax.set_title(r"reach collapses $\propto$ distance to pole: $r=\sqrt{\tau}\,(1-x_0)$")
ax.legend(fontsize=9, loc="lower right"); fig.tight_layout()
fig.savefig("figures/ex2_collapse.png", dpi=120); plt.close(fig)


def plot_2d(name, title, draw_box=False, note=None, box_label="per-axis box (corners overshoot)"):
    X, Y, R, T = grid(f"{name}.csv")
    p = params.get(name)
    tau, f0 = float(p["tau"]), float(p["f0"])
    budget = tau * abs(f0)  # absolute error budget; f0 != 1 here so this matters
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    cf = ax.contourf(X, Y, np.abs(R), levels=25, cmap="magma")
    plt.colorbar(cf, ax=ax, label="|real error|")
    # estimated trusted region boundary (is_trusted) vs the real budget level set
    ax.contour(X, Y, T, levels=[0.5], colors="cyan", linewidths=2)
    ax.contour(X, Y, np.abs(R), levels=[budget], colors="lime",
               linewidths=1.5, linestyles="--")
    if draw_box:
        rx, ry = float(p["rx"]), float(p["ry"])
        if np.isfinite(rx) and np.isfinite(ry):
            ax.add_patch(Rectangle((-rx, -ry), 2 * rx, 2 * ry, fill=False, ec="white",
                                   ls=":", lw=1.5))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="cyan", lw=2,
                      label="cyan: predicted trustworthy (is_trusted, from jet)"),
               Line2D([], [], color="lime", ls="--",
                      label=rf"green: actually trustworthy (truth), $|E|=\tau|f_0|={budget:.3g}$")]
    if draw_box:
        handles.append(Line2D([], [], color="white", ls=":", label=box_label))
    # legend in the bottom margin so it never covers the error field
    fig.legend(handles=handles, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=1, frameon=False)
    ax.set_xlabel("$h_x$"); ax.set_ylabel("$h_y$"); ax.set_aspect("equal")
    ax.set_title(rf"$f_0={f0:g}$,  budget $\tau|f_0|={budget:.3g}$", fontsize=9)
    fig.suptitle(title, fontsize=11)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color="white",
                va="bottom")
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(f"figures/{name}.png", dpi=120); plt.close(fig)


# ---- Example 3: pure quadratic -- estimate exact, trust = exact ellipse ----
plot_2d("ex3", "Example 3 — $1+2x-y+x^2+xy+y^2$: order-2 estimate EXACT (cyan == green)",
        note="no order-3+ terms exist, so the estimate equals the real error\n"
             "(residual ~1e-16) — cyan lands exactly on green.")
# ---- Example 4: coupling -- tilted region, box corners overshoot ----
plot_2d("ex4", "Example 4 — $\\exp(x)+\\exp(y)+0.5xy$: coupling tilts the ellipse, box corners overshoot",
        draw_box=True,
        note="cyan & green gap = order-3+ error the order-2 estimate omits;\n"
             "cyan sits outside green where the estimate is optimistic.")

# ---- Example 5: the model_order trade-off (linear vs quadratic, one jet) ----
rows = csv_rows("ex5.csv")
h = col(rows, "h"); truth = col(rows, "truth")
s1 = col(rows, "surr1"); s2 = col(rows, "surr2")
r1 = col(rows, "real1"); e1 = col(rows, "est1"); r2 = col(rows, "real2"); e2 = col(rows, "est2")
reach1 = float(rows[0]["reach1"]); reach2 = float(rows[0]["reach2"]); budget = float(rows[0]["budget"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].plot(h, truth, "k", lw=2, label="truth $e^h$")
ax[0].plot(h, s1, "C0--", lw=2, label="linear surrogate (model_order=1)")
ax[0].plot(h, s2, "C4-.", lw=2, label="quadratic surrogate (model_order=2)")
ax[0].axvspan(-reach1, reach1, color="C0", alpha=0.12)
ax[0].axvspan(-reach2, reach2, color="C4", alpha=0.10)
ax[0].set_xlabel("step $h$"); ax[0].set_ylabel("value")
ax[0].set_title("same jet, two certified models (shaded = trusted interval)")
ax[1].plot(h, np.abs(r1), "C0", lw=2, label=r"linear $|$real$|$")
ax[1].plot(h, np.abs(e1), "C0:", lw=1.5, label=r"linear $|$est$|=\frac{1}{2}h^2$")
ax[1].plot(h, np.abs(r2), "C4", lw=2, label=r"quad $|$real$|$")
ax[1].plot(h, np.abs(e2), "C4:", lw=1.5, label=r"quad $|$est$|=\frac{1}{6}|h|^3$")
ax[1].axhline(budget, color="C3", ls=":", lw=2, label=rf"budget $={budget:.3g}$")
ax[1].axvline(reach1, color="C0", lw=2); ax[1].axvline(-reach1, color="C0", lw=2)
ax[1].axvline(reach2, color="C4", lw=2); ax[1].axvline(-reach2, color="C4", lw=2)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 1)
ax[1].set_xlabel("step $h$"); ax[1].set_ylabel("absolute error")
ax[1].set_title(f"trusting the quadratic grows reach {reach1:.3f} $\\to$ {reach2:.3f}")
h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
fig.legend(h0, l0, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=1, fontsize=8, frameon=False)
fig.legend(h1, l1, loc="lower right", bbox_to_anchor=(0.97, 0.0), ncol=2, fontsize=7, frameon=False)
fig.suptitle(r"Example 5 — model_order trade-off: linear vs quadratic from one $\langle1,3\rangle$ jet",
             fontsize=12)
fig.tight_layout(rect=[0, 0.13, 1, 0.96]); fig.savefig("figures/ex5.png", dpi=120); plt.close(fig)

# ---- Example 6: a vanishing leading term fools the estimate (2 + sin x) ----
rows = csv_rows("ex6.csv")
h = col(rows, "h"); truth = col(rows, "truth"); surr = col(rows, "surr")
real = col(rows, "real_err"); ef = col(rows, "est_fixed")
reach = float(rows[0]["reach_fixed"]); budget = float(rows[0]["budget"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].plot(h, truth, "k", lw=2, label=r"truth $2+\sin h$")
ax[0].plot(h, surr, "C0--", lw=2, label="linear surrogate $2+h$")
ax[0].axvspan(-reach, reach, color="C2", alpha=0.15, label=f"honest trusted $|h|<{reach:.3f}$")
ax[0].set_xlabel("step $h$"); ax[0].set_ylabel("value")
ax[0].set_title(r"$2+\sin x$: odd function has no $h^2$ term")
ax[1].plot(h, np.abs(real), "k", lw=2, label=r"$|$real error$|\approx|h^3/6|$")
ax[1].plot(h, np.abs(ef), "C1--", lw=2, label=r"cubic est $|c_3h^3|$ (otinum$\langle1,3\rangle$)")
ax[1].axhline(budget, color="C3", ls=":", lw=2, label=rf"budget $={budget:.3g}$")
ax[1].axvline(reach, color="C2", lw=2, label=rf"honest reach $={reach:.3f}$")
ax[1].axvline(-reach, color="C2", lw=2)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-7, 1)
ax[1].set_xlabel("step $h$"); ax[1].set_ylabel("absolute error")
ax[1].set_title(r"naive order-2 estimate $\approx 0$ $\to$ reports $\infty$ reach (DANGER)")
ax[1].text(0.5, 0.05, "linear $\\langle1,2\\rangle$ order-2 band $\\approx 0$ (machine eps)\n"
                      "$\\to$ is_trusted TRUE for every $h$ (wrong)", transform=ax[1].transAxes,
           ha="center", fontsize=8, color="C3")
h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
fig.legend(h0, l0, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=1, fontsize=8, frameon=False)
fig.legend(h1, l1, loc="lower right", bbox_to_anchor=(0.97, 0.0), ncol=1, fontsize=8, frameon=False)
fig.suptitle("Example 6 — a vanishing leading term fools the estimate", fontsize=12)
fig.tight_layout(rect=[0, 0.15, 1, 0.96]); fig.savefig("figures/ex6.png", dpi=120); plt.close(fig)

# ---- Example 7: error_sensitivity as a corrective control step ----
X, Y, R, T = grid("ex7.csv")
p = params["ex7"]; tau = float(p["tau"]); f0 = float(p["f0"]); budget = tau * abs(f0)
path = csv_rows("ex7_path.csv")
px = col(path, "hx"); py = col(path, "hy")
fig, ax = plt.subplots(figsize=(6.8, 6.4))
cf = ax.contourf(X, Y, np.abs(R), levels=25, cmap="magma"); plt.colorbar(cf, ax=ax, label="|error|")
ax.contour(X, Y, T, levels=[0.5], colors="cyan", linewidths=2)
ax.plot(px, py, "o-", color="white", ms=4, lw=1.5)
for i in range(len(px) - 1):  # draw each descent step as an arrow
    ax.annotate("", xy=(px[i + 1], py[i + 1]), xytext=(px[i], py[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
ax.plot(px[0], py[0], "o", color="red", ms=10)
ax.plot(px[-1], py[-1], "o", color="lime", ms=10)
ax.set_xlabel("$h_x$"); ax.set_ylabel("$h_y$"); ax.set_aspect("equal")
ax.set_title(r"descend $-\,$sign$(E)\,\nabla E$ (error_sensitivity) back under budget")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color="cyan", lw=2, label="trust boundary (is_trusted)"),
           Line2D([], [], marker="o", color="red", ls="", label="start (out of trust)"),
           Line2D([], [], marker="o", color="lime", ls="", label="corrected (trusted)")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8,
           frameon=False)
fig.suptitle("Example 7 — error_sensitivity as a corrective control step", fontsize=11)
fig.tight_layout(rect=[0, 0.08, 1, 0.95]); fig.savefig("figures/ex7.png", dpi=120); plt.close(fig)

# ---- Example 8: anisotropy (eccentric trust sliver) ----
plot_2d("ex8", "Example 8 — $1+x^2+100y^2$: anisotropic, $y$ 10$\\times$ stiffer (thin trust sliver)",
        note="reach$_x$/reach$_y$ = 10; one isotropic radius wastes the loose $x$ axis.")
# ---- Example 9: saddle (trust opens along the diagonals) ----
plot_2d("ex9", "Example 9 — $1+x^2-y^2$ saddle: trust opens along the diagonals",
        draw_box=True, box_label="per-axis box (here corners are SAFE, $E=0$)",
        note="$E=h_x^2-h_y^2=0$ on $h_x=\\pm h_y$ $\\to$ infinite diagonal reach;\n"
             "axis-aligned box can't express it (here the box corners are SAFE).")

print("wrote figures/ex1.png .. ex9.png")
