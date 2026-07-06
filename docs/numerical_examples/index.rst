Numerical Examples
==================

These pages collect the library's worked numerical examples, in increasing
order of realism: closed-form functions where every derivative can be checked
against an exact formula, a finite-element heat equation where sensitivities
are validated against finite differences, and three applications that use the
jet from a single PDE solve as a local surrogate model -- parameter
perturbation, uncertainty quantification, and certified reuse in a
digital-twin setting.

Two themes run through all of them:

* **Exactness.** OTI derivatives are exact to floating-point roundoff -- there
  is no step-size parameter and no truncation error in the derivatives
  themselves. The closed-form examples measure this directly; the PDE examples
  show the residual difference against finite differences is the *finite
  difference's* truncation error, not OTI's.
* **One evaluation, many answers.** A single overloaded evaluation with
  ``oti::otinum<M, N>`` returns the value and every mixed partial derivative
  through total order ``N``. The later examples exploit the same coefficients
  a second time, as a Taylor surrogate that answers "what if the parameters
  change?" without re-solving.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Example
     - OTI shape
     - What it demonstrates
   * - :doc:`one_dimensional`
     - ``otinum<1, 3>``
     - value + first three derivatives from one evaluation; local Taylor model
   * - :doc:`two_dimensional`
     - ``otinum<2, 3>``
     - all mixed partials to third order, exact to machine precision
   * - :doc:`heat_equation`
     - ``otinum<3, 1>``
     - PDE parameter sensitivities, validated against central finite differences
   * - :doc:`surrogate`
     - ``otinum<3, 1>``
     - Taylor surrogate at perturbed parameters vs a true re-solve
   * - :doc:`uq_max_temperature`
     - ``otinum<3, 2>``
     - moment propagation from one solve vs 40,000-sample Monte Carlo
   * - :doc:`digital_twin`
     - ``otinum<3, 2>``
     - certified surrogate reuse under parameter drift (validity gate)

The closed-form examples run through the Python bindings
(:doc:`../tutorials/python_bindings`); their sources are in
``examples/python/``. The heat-equation examples are C++ (Kokkos) programs in
the research harness on the
`heat_equation fork <https://github.com/Samm-Py/heat_equation/tree/oti-analysis-and-benchmarks>`_;
a minimal, PR-ready version of the same conversion is described in
:doc:`../tutorials/integration`.

.. toctree::
   :hidden:
   :maxdepth: 1

   one_dimensional
   two_dimensional
   heat_equation
   surrogate
   uq_max_temperature
   digital_twin
