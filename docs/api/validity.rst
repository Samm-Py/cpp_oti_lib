Validity Analysis
=================

This page is generated from ``include/otinum/validity.hpp``. The
``oti::validity`` namespace treats a jet as a local Taylor surrogate in the
seeded variables and answers how far that surrogate can be trusted. It is a
separate include — deliberately not part of the ``otinum.hpp`` umbrella:

.. code-block:: cpp

   #include "otinum/validity.hpp"

:doc:`../tutorials/validity` is the worked introduction; this page holds the
contracts.

The Model-Order Contract
------------------------

Every primitive shares one ``model_order`` parameter, ``m`` (default
``N - 1``). You *certify* the order-``m`` surrogate; its leading truncation
error is the order-``m + 1`` term, so the jet must be computed at
``N >= m + 1`` for that term to exist in storage. The prediction
(``evaluate``) and the certification (``truncation_error``, ``is_trusted``,
``validity_radius``, ``error_sensitivity``) take the same ``m``, so they
always describe the same model.

Computing more than one order above the model is never wasted: the error
primitives fold **every** stored order from ``m + 1`` through ``N``. A linear
model certified from an order-3 jet uses both the quadratic and the cubic
coefficients in its error estimate, giving a sharper bound than an order-2
jet could.

The Primitives
--------------

All functions take the jet, and most take a step ``h`` — an
``oti::detail::array<Coeff, M>`` (``std::array`` on host builds,
``Kokkos::Array`` under ``OTI_ENABLE_KOKKOS``) holding one displacement per
seeded variable, in that variable's own units.

``evaluate(jet, h, m = N-1) -> Coeff``
   The surrogate prediction: the sum of the jet's monomials of total order
   ``<= m`` evaluated at ``h``. Stored coefficients are already Taylor
   coefficients, so no factorials enter. May also be called with ``m = N``
   for the full-jet prediction — more accurate, but with no stored band
   above it, uncertifiable from the same jet.

``truncation_error(jet, h, m = N-1) -> Coeff``
   The signed error estimate of the order-``m`` prediction at ``h``: the sum
   of every stored band above the model,
   ``sum over m+1 <= |beta| <= N of c_beta * h^beta``. Mixed (coupling)
   coefficients participate like any others. Requires ``m < N``.

``is_trusted(jet, h, tau, tau_abs = 0, m = N-1) -> bool``
   The trust predicate: ``|truncation_error| <= tau_abs + tau * |f|`` with
   ``f`` the jet's real coefficient. ``tau`` is a relative tolerance;
   ``tau_abs`` is an absolute floor in the quantity's own units. The floor
   matters for signed quantities whose value passes through zero
   (temperature differences, residuals, fluxes) — there a purely relative
   budget collapses to zero and everything reports untrusted, however exact
   the surrogate is.

``validity_radius(jet, tau, tau_abs = 0, m = N-1) -> array<Coeff, M>``
   Per-variable reach: the largest single-axis step ``r_i`` along each
   variable for which the model stays within the budget. With a single
   stored error band this is a closed form,
   ``r_i = (budget / |c_(m+1)e_i|)^(1/(m+1))``; with several bands it is
   found by an allocation-free bracket-and-bisection. A variable whose
   pure-axis error vanishes entirely (the model is exact along that axis)
   reports ``+inf``.

``error_sensitivity(jet, h, m = N-1) -> array<Coeff, M>``
   The gradient of the truncation error with respect to ``h`` — the
   steepest-descent direction for pulling an untrusted step back under
   tolerance. Because it is a gradient, interaction (mixed-partial) terms
   are distributed across their variables with no ad-hoc convention.

Radii Are Per-Axis Statements, Not A Box
----------------------------------------

``validity_radius`` answers ``M`` independent one-dimensional questions. The
corner ``(r_0, r_1, ...)`` moves every variable to its individual limit at
once, and the pure terms plus coupling terms stack — combined steps routinely
bust the budget there. The exact predicate for any multi-axis step is always
``is_trusted(jet, h, tau)``; treat the radii as the human-readable summary
and the tilt of the true trusted region (visualized by the worked examples in
``examples/validity/``) as the reason not to promote them to box sides.

Host And Device
---------------

Everything on this page is ``OTI_CONSTEXPR_FUNCTION``: host- and
device-callable, allocation-free, exception-free. The implementations fold
over compile-time index sequences (the same technique as the arithmetic
kernels), so on GPU they compile to straight-line register code with no
local-memory table traffic — measured at 0 bytes of stack frame across the
shapes in the test suite. They are safe to call inside a Kokkos kernel per
element, whether in a post-solve trust-map pass or in-loop control logic.

Generated Reference
-------------------

.. doxygennamespace:: oti::validity
   :content-only:
   :undoc-members:
