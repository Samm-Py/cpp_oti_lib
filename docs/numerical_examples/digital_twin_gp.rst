Digital Twin II: A Gaussian-Process Twin From Jets
==================================================

The :doc:`digital_twin` twin serves each query from the *nearest* certified
anchor in its atlas -- every prediction comes from one jet at a time. This
page takes the same jets one step further: **fuse all anchors into a single
global surrogate** -- a derivative-enhanced Gaussian process, using `JetGP
<https://github.com/Samm-Py/jetgp>`_ (`documentation
<https://samm-py.github.io/jetgp/>`_) -- over all **three** parameters at
once, so every anchor ever solved informs every prediction, the posterior
interpolates *between* anchors instead of expanding around one, and its
standard deviation supplies the reuse gate. The payoff is measured in the only
units that matter: PDE solves.

Jets Are Training Data
----------------------

One ``otinum<3, 2>`` solve returns the QoI's value, gradient, and full Hessian
(mixed partials included) with respect to the three parameters -- **10 exact
observations from one PDE solve**, with none of the step-size noise that
finite-difference derivatives would inject into a GP's covariance structure.
JetGP's ``DEGP`` module consumes exactly this: coordinate-aligned partial
derivatives of arbitrary order, declared as derivative multi-indices:

.. code-block:: python

   from jetgp.full_degp.degp import degp

   # one entry per observed derivative: [variable, order] factors
   der_indices = [
       [[[1, 1]], [[2, 1]], [[3, 1]]],                    # gradient
       [[[1, 2]], [[1, 1], [2, 1]], [[1, 1], [3, 1]],     # Hessian,
        [[2, 2]], [[2, 1], [3, 1]], [[3, 2]]],            # mixed included
   ]
   model = degp(X_anchors, [q, dq_da, dq_dA, dq_ds, ...],
                n_order=2, n_bases=3, der_indices=der_indices,
                derivative_locations=locations, kernel="SE",
                kernel_type="anisotropic")

Because the order-2 jet contains the order-1 jet, a single exported anchor
bank serves every variant below -- the comparison isolates the *information
content per solve*, nothing else.

What A Derivative-Enhanced Anchor Buys
--------------------------------------

Four surrogates are trained on the **same** nested Halton sequence of anchor
points in the box and measured against **400 Monte Carlo truth solves**
(genuine PDE re-solves at random parameter points -- affordable here, which is
exactly what makes the claim auditable):

.. image:: ../_static/numerical_examples/gp_twin_convergence.png
   :alt: RMS error vs number of anchor solves for the four surrogates
   :width: 95%
   :align: center

.. list-table:: Anchor PDE solves needed to reach the tolerance (1.5e-3, ~1% of the nominal QoI)
   :header-rows: 1
   :widths: 40 20 40

   * - Surrogate
     - Solves
     - Observations per solve
   * - value-only GP
     - 16
     - 1
   * - order-1 jet GP
     - 4
     - 4 (value + gradient)
   * - order-2 jet GP
     - **2**
     - 10 (+ full Hessian)
   * - nearest-anchor Taylor
     - 2
     - 10, used locally

That table is what "the derivatives buy you," in units everyone accepts: PDE
solves. Three structural readings from the curves:

* **The jet advantage is largest exactly where it matters** -- at low anchor
  counts, the expensive-simulation regime. At 2 anchors the order-2 jet GP is
  already 58x more accurate than the value-only GP; at 16 anchors it is three
  orders of magnitude ahead.
* **Local vs global use of the same data.** The nearest-anchor Taylor
  evaluation (jet data without probabilistic fusion) matches the jet GP at 2
  anchors, then *plateaus* near 2e-5: a local model's error is set by the
  distance to its anchor, so it improves only as anchors densify. The GPs fuse
  all anchors into one global posterior and keep converging -- the jet GPs
  reach the ~6e-8 floor, and even the value-only GP eventually overtakes
  Taylor.
* **Cost accounting stays favorable.** An ``otinum<3, 2>`` solve carries 10
  coefficients per node and costs a small multiple of a plain solve -- well
  under the 8x anchor-count reduction (16 -> 2), and each anchor's jet is
  computed once and reused forever.

The Twin Loop
-------------

The operational scenario from :doc:`digital_twin`, now in 3-D: all three
parameters drift along a closed path through the box, traversed 1.5 times over
200 updates -- so the last third of the queries returns to parameter territory
from the first traversal. Four twins serve the identical query stream:

* **Taylor atlas** -- the previous page's twin: every anchor jet is kept, each
  query is served by the nearest one under the order-2 validity gate.
* **value / order-1 / order-2 GP twins** -- every anchor is training data for
  one global posterior; re-solve when its standard deviation at the query
  exceeds the same tolerance. Three levels of information per solve.

.. image:: ../_static/numerical_examples/gp_twin_loop.png
   :alt: cumulative solves and error traces for the four twins along the drift path
   :width: 100%
   :align: center

.. list-table::
   :header-rows: 1
   :widths: 34 18 24 24

   * - Twin
     - Solves
     - Max abs. error
     - Mean abs. error
   * - Taylor atlas
     - 12
     - 3.3e-3
     - 3.3e-4
   * - value GP
     - 12
     - 3.2e-3
     - 3.1e-4
   * - order-1 jet GP
     - 6
     - 4.3e-3
     - 3.4e-4
   * - order-2 jet GP
     - **5**
     - **1.3e-3**
     - **1.0e-4**

Returning is cheap for every twin here -- after the path repeats (dashed
line), all four run essentially flat, since the second traversal is covered by
anchors from the first. The differences are set earlier, in *new* territory,
and they isolate fusion cleanly. The Taylor atlas consults one jet at a time,
so it must plant anchors densely enough that every query is near one: 12
solves. The value GP fuses globally but each solve carries a single number:
also 12. Fusing the *jets* is what breaks below -- all ten derivatives of
every anchor shape the posterior everywhere, so far fewer anchors make the
whole path confident: 6 solves at order 1, **5 at order 2**. The order-2 jet
twin serves 200 updates with 5 solves and is the only variant whose worst-case
error stays essentially at the tolerance.

One honest caveat, visible in the bottom panel: the GP gate is
**probabilistic**, not certified. A one-standard-deviation criterion admits
occasional excursions slightly above the tolerance (all twins show a few,
worst case ~2.8x for the order-1 GP); a stricter ``k * std`` gate buys margin
at the cost of a few more solves. This is the structural trade against
:doc:`digital_twin`: the validity gate certifies a hard truncation bound,
anchor by anchor; the GP gate is calibrated-statistical, but fusion is what
lets it cover the same path with fewer anchors. Which to prefer is an
application decision -- and since both consume the same jets, switching
between them (or nesting them) requires no new PDE machinery.

Reproducing It
--------------

The experiment is deliberately split so no PDE solver is needed to rerun the
GP side:

* **The jet bank** -- Halton anchors with full jets, the Monte Carlo truth
  set, and the drift path (truth + jets) -- is exported by ``uq_gp_bank.cpp``
  on the `oti-analysis-and-benchmarks branch
  <https://github.com/Samm-Py/heat_equation/tree/oti-analysis-and-benchmarks>`_
  of the heat-equation fork (664 solves, ~30 s on a laptop CPU), and the CSVs
  are committed in this repository under ``examples/python/data/gp_bank/``.
* **The experiments** are ``examples/python/digital_twin_gp.py``, which needs
  `JetGP <https://github.com/Samm-Py/jetgp>`_ and its conda environment (see
  the JetGP README; it builds a patched otilib backend). The figures above are
  the committed output of that script, so the documentation itself has no
  JetGP dependency.
