cpp_oti_lib Documentation
=========================

``cpp_oti_lib`` is a header-only C++ library for static OTI numbers stored as
truncated multivariate Taylor polynomials. The primary type is
``oti::otinum<M, N, Coeff = double>``, where ``M`` is the number of variables,
``N`` is the maximum total derivative order, and ``Coeff`` selects the
floating-point coefficient type.

Automatic differentiation is a useful tool for computing sensitivities, but
introducing derivative calculations into existing C++ simulation codes can
require substantial source-code modification. ``cpp_oti_lib`` provides a compact
header-only implementation of order-truncated imaginary (OTI) numbers for
hypercomplex automatic differentiation.

An OTI algebra ``OTI_m^n`` propagates derivative information with respect to
``m`` independent variables through total order ``n``. One overloaded model
evaluation can return the function value and the corresponding sensitivities.
The library provides overloaded arithmetic and elementary functions so many
numerical kernels can be differentiated by changing selected scalar types
rather than rewriting the kernel logic.

The implementation includes focused C++ unit tests, Python examples for
visualization and experimentation, single- and double-precision coefficient
storage, and an optional Kokkos-enabled path for CPU and GPU backends. Together,
these pieces provide a practical workflow for obtaining sensitivities from C++
simulation codes.

Suggested Reading Path
----------------------

If you are new to the project:

* Start with :doc:`readme` to compile and run the smallest complete C++
  example.
* Continue to :doc:`installation` when you want local tests, Python bindings,
  Kokkos builds, or a fresh documentation build.
* Work through :doc:`tutorials/basic_usage` for a fuller derivative check.
* Use :doc:`api/index` when you need coefficient semantics, layout details, or
  generated C++ API reference pages.

Reports
-------

When documentation deployment is enabled, the CI documentation build publishes
the coverage report with the hosted site. Local documentation builds can use
the same path after generating coverage:

.. raw:: html

   <p><a href="generated/coverage/index.html">Open the coverage report</a></p>

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   readme
   installation

.. toctree::
   :maxdepth: 2
   :caption: Reports

   coverage

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/basic_usage
   tutorials/float_coefficients
   tutorials/directional_derivatives
   tutorials/kokkos_cpu
   tutorials/kokkos_gpu
   tutorials/soa_layout
   tutorials/plotting

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Status
------

The C++ scalar library is header-only. Python bindings and Kokkos support are
optional build paths layered on top of the same headers.
