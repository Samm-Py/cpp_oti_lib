Trusting A Jet: Validity Analysis
=================================

A jet is more than a bundle of derivatives — it is a local Taylor *surrogate*
of the function it came from. Evaluate the polynomial at a step ``h`` away
from the expansion point and you get a prediction of ``f(x0 + h)`` with no
re-evaluation of ``f``. That is enormously useful (a simulation evaluated
once answers "what if" questions for free), but only inside the region where
the truncated polynomial still tracks the function. This tutorial introduces
``otinum/validity.hpp``, which answers the question every surrogate user has
to ask: *how far can I trust this?*

The Idea: Compute Above The Model You Trust
-------------------------------------------

An order-``N`` jet cannot bound its own truncation error — the leading error
of an order-``N`` model is the order-``N+1`` term, which the jet does not
store. The library's convention follows directly: **compute the jet at least
one order above the model you intend to certify**. An ``otinum<M, 2>`` jet
certifies the *linear* surrogate, spending its second-order coefficients as
the error estimate; an ``otinum<M, 3>`` jet certifies the *quadratic*
surrogate by default — or, still, the linear one.

That last option matters: the model order and the computed order are
independent choices. A linear surrogate certified from an order-3 jet uses
*both* the second- and third-order coefficients to quantify its error —
``truncation_error`` folds every stored order above the model, so each extra
computed order sharpens the estimate rather than going to waste. Every
function on this page shares one ``model_order`` argument (defaulting to
``N - 1``) so the model you predict with and the model you certify are always
the same.

Program
-------

The same source is available in the repository as
``examples/validity_intro.cpp``.

.. literalinclude:: ../../examples/validity_intro.cpp
   :language: cpp

Compile And Run
---------------

From the repository root:

.. code-block:: console

   c++ -std=c++17 -I include examples/validity_intro.cpp -o /tmp/validity_intro
   /tmp/validity_intro

Output
------

.. code-block:: text

   value                 = 2
   linear prediction     = 2.1
   estimated error       = 0.035
   budget (tau*|f|)      = 0.1
   trusted at h?         = yes
   per-axis reach        = (0.447214, 0.447214)
   trusted at corner?    = no
   error at corner       = 0.3
   error sensitivity     = (0.2, -0.05)

How It Works
------------

**The jet as a surrogate.** ``f = exp(x) + exp(y) + x*y/2`` expanded at
``(0, 0)`` gives value ``2``, gradient ``(1, 1)``, and stored second-order
coefficients ``c20 = 0.5``, ``c11 = 0.5``, ``c02 = 0.5`` (stored coefficients
are Taylor coefficients — no factorials to manage).

**Prediction.** ``evaluate(f, h)`` sums the model's monomials at
``h = (0.3, -0.2)``: the linear surrogate gives
``2 + 1*0.3 + 1*(-0.2) = 2.1``.

**Error estimate.** ``truncation_error(f, h)`` sums every *stored* order above
the model — here the single order-2 band:
``0.5*0.09 + 0.5*(0.3 * -0.2) + 0.5*0.04 = 0.035``. Note the middle term: the
mixed coefficient participates, so parameter *coupling* is captured
automatically.

**Trust check.** ``is_trusted(f, h, tau)`` compares that estimate against the
budget ``tau * |f|``: ``0.035 <= 0.1``, so the linear prediction at this step
is certified within 5%. For signed quantities whose value can pass through
zero (a temperature *difference*, a residual), a purely relative budget
collapses; pass the optional absolute floor, ``is_trusted(f, h, tau,
tau_abs)``, and the budget becomes ``tau_abs + tau * |f|``.

**Reach.** ``validity_radius(f, tau)`` reports, per variable, the largest
single-axis step that stays within budget. With one error band it is the
closed form ``sqrt(budget / c_pure) = sqrt(0.1 / 0.5) = 0.447``; with more
stored bands the library solves it by bisection internally — the call does
not change.

**The corner trap.** The reaches are *single-axis* statements, not the sides
of a safe box. Stepping to ``(r0, r1)`` moves both variables to their
individual limits simultaneously, and the pure terms plus the coupling term
stack up: the error there is ``0.3`` — three times the budget. The exact
predicate for any combined step is always ``is_trusted``; the radii are the
human-readable summary.

**Blame.** When a step is not trusted, ``error_sensitivity(f, h)`` returns
the gradient of the truncation error with respect to ``h`` — the
steepest-descent direction for pulling the step back under tolerance. At our
trusted ``h`` it is ``(0.2, -0.05)``: variable 0 currently drives the error
budget four times harder than variable 1, coupling included.

Choosing The Model Order Independently
--------------------------------------

Nothing above is specific to linear models, and the model order is not tied
to the computed order. From one order-3 jet, the same calls support three
different uses:

.. code-block:: cpp

   using T3 = oti::otinum<2, 3>;
   // ... build the jet ...

   // Quadratic model (the default, N-1 = 2): cubic band is the error.
   double pred_q = v::evaluate(jet, h);
   bool ok_q     = v::is_trusted(jet, h, tau, 0.0);

   // LINEAR model from the same jet (model_order = 1): the quadratic AND
   // cubic bands BOTH fold into the error estimate -- a sharper, less
   // optimistic bound than an order-2 jet could give the same model.
   double pred_l = v::evaluate(jet, h, 1);
   bool ok_l     = v::is_trusted(jet, h, tau, 0.0, 1);

   // Full-jet prediction (model_order = 3): the most accurate number the
   // jet can produce -- but no stored band remains above it, so no
   // same-jet error bound exists for it.
   double pred_full = v::evaluate(jet, h, 3);

The middle case is the general pattern: computing extra orders beyond
``model_order + 1`` is never wasted, because ``truncation_error``,
``is_trusted``, ``validity_radius``, and ``error_sensitivity`` all fold every
stored order above the model into the error.

Where This Runs
---------------

Everything in ``validity.hpp`` is allocation-free and device-callable
(``OTI_CONSTEXPR_FUNCTION``): the same primitives work in ordinary host code,
in a post-solve Kokkos ``parallel_for`` over a stored field of jets (a
per-point "trust map"), or inside a live GPU kernel gating whether to reuse a
surrogate or re-run a simulation. The step ``h`` is an
``oti::detail::array<Coeff, M>`` — ``std::array`` on host builds and
``Kokkos::Array`` in Kokkos builds.

A set of nine analytic, hand-checkable worked examples ships in
``examples/validity/`` with plotting scripts that visualize trust regions,
including the tilted-ellipse coupling picture behind the corner trap above.

Where To Go Next
----------------

* :doc:`../api/validity` documents the five primitives, the shared
  ``model_order`` contract, and the budget semantics in full.
* :doc:`directional_derivatives` — the validity step ``h`` composes naturally
  with directional seeding.
* :doc:`kokkos_gpu` covers the device build these primitives are designed for.
