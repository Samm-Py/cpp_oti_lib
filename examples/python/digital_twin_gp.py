"""GP digital twin from OTI jets: what a derivative-enhanced anchor buys.

Runs two experiments against the precomputed jet bank in ``data/gp_bank/``
(exported by ``uq_gp_bank.cpp`` on the heat-equation analysis fork; see the
"Digital Twin II" page of the documentation), comparing four surrogates of the
final-time sensor temperature over the 3-D parameter box (alpha, A, sigma):

  value GP      one observation per anchor solve (the QoI value)
  order-1 jetGP + exact gradient           (4 observations per solve)
  order-2 jetGP + exact Hessian, mixed too (10 observations per solve)
  Taylor        second-order Taylor evaluation from the NEAREST anchor jet
                (jet data without probabilistic fusion)

Experiment 1 (convergence): the four surrogates trained on the same nested
Halton anchor sequence, errors measured against 400 Monte Carlo truth solves;
reports the number of anchor solves each needs to reach the error tolerance.

Experiment 2 (twin loop): a drifting-parameter query stream (closed path,
traversed 1.5x so late queries REVISIT early territory), served by five twins
that separate two effects -- memory and fusion. GP twins re-solve when the
posterior std exceeds the tolerance and remember every anchor. Two Taylor
twins use the identical order-2 gate: one holds only the LATEST anchor (the
previous docs page's design), one keeps an ATLAS of all anchors and serves
each query from the nearest -- memory without fusion.

Requires jetgp (https://github.com/Samm-Py/jetgp) and its environment; the
figures committed under figures/digital_twin_gp/ were produced by this script,
so the documentation does not require jetgp to build.

Run:  python digital_twin_gp.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from jetgp.full_degp.degp import degp

HERE = Path(__file__).resolve().parent
BANK = HERE / "data" / "gp_bank"
FIGURE_DIR = HERE / "figures" / "digital_twin_gp"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260707
TOL = 1.5e-3          # absolute QoI tolerance (~1% of the nominal QoI 0.154)
ANCHOR_COUNTS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
JADE = dict(optimizer="jade", pop_size=40, n_generations=12)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# ---------------------------------------------------------------------------
# Bank loading. Column order of the jet blocks matches uq_gp_bank.cpp:
# q, dq/da, dq/dA, dq/ds, d2q/da2, d2q/dadA, d2q/dads, d2q/dA2, d2q/dAds, d2q/ds2
# ---------------------------------------------------------------------------

JET_COLS = ["q", "dq_da", "dq_dA", "dq_ds",
            "d2q_da2", "d2q_dadA", "d2q_dads", "d2q_dA2", "d2q_dAds", "d2q_ds2"]

# jetgp derivative multi-indices for the same order (1-indexed variables)
DER_INDICES_O1 = [[[[1, 1]], [[2, 1]], [[3, 1]]]]
DER_INDICES_O2 = DER_INDICES_O1 + [[[[1, 2]], [[1, 1], [2, 1]], [[1, 1], [3, 1]],
                                    [[2, 2]], [[2, 1], [3, 1]], [[3, 2]]]]


def load_csv(name):
    data = np.genfromtxt(BANK / name, delimiter=",", names=True)
    return data


ANCHORS = load_csv("bank_anchors.csv")
MC = load_csv("bank_mc_truth.csv")
DRIFT = load_csv("bank_drift_truth.csv")

X_MC = np.column_stack([MC["alpha"], MC["amplitude"], MC["sigma"]])
Q_MC = MC["q_true"]
X_DRIFT = np.column_stack([DRIFT["alpha"], DRIFT["amplitude"], DRIFT["sigma"]])
Q_DRIFT = DRIFT["q_true"]

# Normalize inputs to the unit box for GP length scales and Taylor distance;
# derivative observations are rescaled by the chain rule accordingly.
LO = np.array([0.7, 70.0, 0.035])
HI = np.array([1.6, 130.0, 0.065])
SPAN = HI - LO


def unit(x):
    return (x - LO) / SPAN


def jet_rows(rows):
    """Extract (X, value, grad, hess) in UNIT-box coordinates from bank rows."""
    x = unit(np.column_stack([rows["alpha"], rows["amplitude"], rows["sigma"]]))
    q = rows["q"]
    g = np.column_stack([rows["dq_da"], rows["dq_dA"], rows["dq_ds"]]) * SPAN
    h = np.column_stack([
        rows["d2q_da2"] * SPAN[0] * SPAN[0],
        rows["d2q_dadA"] * SPAN[0] * SPAN[1],
        rows["d2q_dads"] * SPAN[0] * SPAN[2],
        rows["d2q_dA2"] * SPAN[1] * SPAN[1],
        rows["d2q_dAds"] * SPAN[1] * SPAN[2],
        rows["d2q_ds2"] * SPAN[2] * SPAN[2],
    ])
    return x, q, g, h


def train_gp(x, q, g=None, h=None):
    """Train a DEGP on value (+ gradient (+ Hessian)) observations."""
    n = len(q)
    y = [q.reshape(-1, 1)]
    if g is None:
        order, der_indices, n_der = 0, None, 0
    elif h is None:
        order, der_indices, n_der = 1, DER_INDICES_O1, 3
        y += [g[:, i].reshape(-1, 1) for i in range(3)]
    else:
        order, der_indices, n_der = 2, DER_INDICES_O2, 9
        y += [g[:, i].reshape(-1, 1) for i in range(3)]
        y += [h[:, i].reshape(-1, 1) for i in range(6)]
    locs = [list(range(n)) for _ in range(n_der)] or None
    model = degp(x, y, n_order=order, n_bases=3, der_indices=der_indices,
                 derivative_locations=locs, normalize=True,
                 kernel="SE", kernel_type="anisotropic")
    np.random.seed(RNG_SEED)
    params = model.optimize_hyperparameters(**JADE)
    return model, params


def gp_predict(model, params, x_test, want_var=False):
    out = model.predict(x_test, params, calc_cov=want_var)
    if want_var:
        mean, var = out[0], out[1]
        return np.asarray(mean).flatten(), np.maximum(np.asarray(var).flatten(), 0.0)
    return np.asarray(out).flatten()


def taylor_predict(x_test, xa, qa, ga, ha):
    """Second-order Taylor evaluation from the NEAREST anchor (unit coords)."""
    pred = np.empty(len(x_test))
    iu, ju = np.triu_indices(3)
    for k, xt in enumerate(x_test):
        d2 = np.sum((xa - xt) ** 2, axis=1)
        j = int(np.argmin(d2))
        dx = xt - xa[j]
        H = np.zeros((3, 3))
        H[iu, ju] = ha[j]
        H = H + np.triu(H, 1).T
        pred[k] = qa[j] + ga[j] @ dx + 0.5 * dx @ H @ dx
    return pred


# ---------------------------------------------------------------------------
# Experiment 1: error vs number of anchor solves (shared Halton sequence)
# ---------------------------------------------------------------------------

def experiment_convergence():
    xa, qa, ga, ha = jet_rows(ANCHORS)
    xt = unit(X_MC)
    results = {"value GP": [], "order-1 jet GP": [], "order-2 jet GP": [],
               "nearest-anchor Taylor": []}
    for k in ANCHOR_COUNTS:
        sl = slice(0, k)
        m, p = train_gp(xa[sl], qa[sl])
        e0 = gp_predict(m, p, xt) - Q_MC
        m, p = train_gp(xa[sl], qa[sl], ga[sl])
        e1 = gp_predict(m, p, xt) - Q_MC
        m, p = train_gp(xa[sl], qa[sl], ga[sl], ha[sl])
        e2 = gp_predict(m, p, xt) - Q_MC
        et = taylor_predict(xt, xa[sl], qa[sl], ga[sl], ha[sl]) - Q_MC
        for key, e in [("value GP", e0), ("order-1 jet GP", e1),
                       ("order-2 jet GP", e2), ("nearest-anchor Taylor", et)]:
            results[key].append((np.sqrt(np.mean(e ** 2)), np.max(np.abs(e))))
        print(f"k={k:3d}  rms: value {results['value GP'][-1][0]:.2e}  "
              f"o1 {results['order-1 jet GP'][-1][0]:.2e}  "
              f"o2 {results['order-2 jet GP'][-1][0]:.2e}  "
              f"taylor {results['nearest-anchor Taylor'][-1][0]:.2e}")
    return results


# ---------------------------------------------------------------------------
# Experiment 2: the twin loop on the drift path
# ---------------------------------------------------------------------------

def drift_jets():
    return jet_rows(DRIFT)


def twin_loop_gp(order):
    """Variance-gated adaptive GP twin: seeded with the first two queries (a
    one-point GP cannot estimate its own scales), then re-solve (and remember)
    whenever the posterior std at the query exceeds TOL."""
    xd, qd, gd, hd = drift_jets()
    anchors = [0, 1]
    preds = np.empty(len(xd))
    solves = np.zeros(len(xd), dtype=bool)
    solves[[0, 1]] = True

    def fit():
        sl = np.array(anchors)
        if order == 0:
            return train_gp(xd[sl], qd[sl])
        if order == 1:
            return train_gp(xd[sl], qd[sl], gd[sl])
        return train_gp(xd[sl], qd[sl], gd[sl], hd[sl])

    model, params = fit()
    for i in range(len(xd)):
        mean, var = gp_predict(model, params, xd[i:i + 1], want_var=True)
        # NaN-robust: anything that is not demonstrably within tolerance
        # triggers a re-solve.
        trusted = np.isfinite(mean[0]) and np.sqrt(var[0]) <= TOL
        if not trusted and i not in anchors:
            anchors.append(i)
            solves[i] = True
            model, params = fit()
            mean, _ = gp_predict(model, params, xd[i:i + 1], want_var=True)
        preds[i] = mean[0]
    return preds, solves


def _taylor_gate_predict(xq, xd, qd, gd, hd, anchor):
    """Linear prediction from one anchor + its order-2 term (the gate signal),
    mirroring oti::validity::is_trusted with an absolute budget."""
    iu, ju = np.triu_indices(3)
    dx = xq - xd[anchor]
    H = np.zeros((3, 3))
    H[iu, ju] = hd[anchor]
    H = H + np.triu(H, 1).T
    second = 0.5 * dx @ H @ dx
    pred = qd[anchor] + gd[anchor] @ dx
    return pred, abs(second)


def twin_loop_taylor():
    """Memoryless Taylor twin (the previous page's design): order-1 model from
    the LATEST anchor only, re-solving when the jet's own order-2 term at the
    offset exceeds TOL."""
    xd, qd, gd, hd = drift_jets()
    preds = np.empty(len(xd))
    solves = np.zeros(len(xd), dtype=bool)
    a = 0
    solves[0] = True
    for i in range(len(xd)):
        pred, second = _taylor_gate_predict(xd[i], xd, qd, gd, hd, a)
        if second > TOL:
            a = i
            solves[i] = True
            pred, _ = _taylor_gate_predict(xd[i], xd, qd, gd, hd, a)
        preds[i] = pred
    return preds, solves


def twin_loop_taylor_atlas():
    """Taylor twin WITH memory: every anchor jet is kept, each query is served
    by the nearest stored anchor under the same order-2 gate. Memory without
    probabilistic fusion -- the control separating what memory fixes (revisits)
    from what the GP's fusion adds (fewer anchors, calibrated uncertainty)."""
    xd, qd, gd, hd = drift_jets()
    preds = np.empty(len(xd))
    solves = np.zeros(len(xd), dtype=bool)
    atlas = [0]
    solves[0] = True
    for i in range(len(xd)):
        d2 = np.sum((xd[atlas] - xd[i]) ** 2, axis=1)
        a = atlas[int(np.argmin(d2))]
        pred, second = _taylor_gate_predict(xd[i], xd, qd, gd, hd, a)
        if second > TOL and i not in atlas:
            atlas.append(i)
            solves[i] = True
            pred, _ = _taylor_gate_predict(xd[i], xd, qd, gd, hd, i)
        preds[i] = pred
    return preds, solves


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

STYLE = {
    "value GP": dict(color="#888888", marker="o"),
    "order-1 jet GP": dict(color="#1f77b4", marker="s"),
    "order-2 jet GP": dict(color="#d62728", marker="^"),
    "nearest-anchor Taylor": dict(color="#2ca02c", marker="d"),
    "Taylor (latest anchor)": dict(color="#2ca02c", marker="d"),
    "Taylor atlas (nearest)": dict(color="#8c564b", marker="v"),
}


def figure_convergence(results):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    reached = {}
    for key, vals in results.items():
        rms = np.array([v[0] for v in vals])
        ax.plot(ANCHOR_COUNTS, rms, label=key, lw=1.8, ms=5, **STYLE[key])
        below = [k for k, r in zip(ANCHOR_COUNTS, rms) if r <= TOL]
        reached[key] = below[0] if below else None
    ax.axhline(TOL, color="k", ls=":", lw=1)
    ax.text(ANCHOR_COUNTS[-1], TOL * 1.15, "tolerance", ha="right", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ANCHOR_COUNTS)
    ax.set_xticklabels([str(k) for k in ANCHOR_COUNTS])
    ax.set_xlabel("number of anchor PDE solves")
    ax.set_ylabel("RMS error vs 400 Monte Carlo truth solves")
    ax.set_title("What a derivative-enhanced anchor buys")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIGURE_DIR / "convergence.png", dpi=220)
    plt.close(fig)
    return reached


def figure_twin_loop(runs):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True, constrained_layout=True)
    n = len(Q_DRIFT)
    for key, (preds, solves) in runs.items():
        ax1.plot(np.arange(n), np.cumsum(solves), lw=1.8,
                 color=STYLE[key]["color"], label=f"{key} ({int(solves.sum())})")
        ax2.semilogy(np.arange(n), np.abs(preds - Q_DRIFT) + 1e-12, lw=1.0,
                     color=STYLE[key]["color"])
    ax1.axvline(n * 2 / 3, color="k", ls="--", lw=0.8)
    ax1.text(n * 2 / 3 + 2, 1, "path repeats", fontsize=9, rotation=90, va="bottom")
    ax2.axvline(n * 2 / 3, color="k", ls="--", lw=0.8)
    ax2.axhline(TOL, color="k", ls=":", lw=1)
    ax2.text(n - 2, TOL * 1.2, "tolerance", ha="right", fontsize=9)
    ax1.set_ylabel("cumulative PDE solves")
    ax1.set_title("Digital-twin loop on the drifting-parameter path")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.set_ylabel("|prediction - truth|")
    ax2.set_xlabel("query index along the drift path")
    fig.savefig(FIGURE_DIR / "twin_loop.png", dpi=220)
    plt.close(fig)


def main():
    print("=== experiment 1: convergence over shared Halton anchors ===")
    results = experiment_convergence()
    reached = figure_convergence(results)
    print("\nanchor solves needed to reach tolerance "
          f"{TOL:.1e} (RMS over the MC set):")
    for key, k in reached.items():
        print(f"  {key:24s}: {k if k is not None else 'not reached at 64'}")

    print("\n=== experiment 2: twin loop on the drift path ===")
    runs = {}
    runs["Taylor (latest anchor)"] = twin_loop_taylor()
    runs["Taylor atlas (nearest)"] = twin_loop_taylor_atlas()
    for key, order in [("value GP", 0), ("order-1 jet GP", 1),
                       ("order-2 jet GP", 2)]:
        runs[key] = twin_loop_gp(order)
    figure_twin_loop(runs)
    print(f"{'twin':24s} {'solves':>7s} {'max |err|':>11s} {'mean |err|':>11s}")
    for key, (preds, solves) in runs.items():
        err = np.abs(preds - Q_DRIFT)
        print(f"{key:24s} {int(solves.sum()):7d} {err.max():11.2e} {err.mean():11.2e}")

    print(f"\nFigures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
