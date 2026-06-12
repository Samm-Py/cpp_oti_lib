Tutorials
=========

These tutorials focus on practical workflows:

* :doc:`basic_usage` verifies first- and second-order derivatives against
  analytic values.
* :doc:`float_coefficients` shows how to use ``float`` storage instead of the
  default ``double`` storage.
* :doc:`directional_derivatives` shows how to seed a line through a larger
  physical input space and compute derivatives along that direction.
* :doc:`cmake_package` builds a separate CMake project against an installed
  copy of the library through ``find_package(otinum)``.
* :doc:`python_bindings` uses the optional Python module: bound types,
  multi-index access, and adding new instantiations.
* :doc:`kokkos_cpu` and :doc:`kokkos_gpu` build the Kokkos smoke test on CPU
  and CUDA backends.
* :doc:`soa_layout` stores arrays of OTI numbers coefficient-major for
  coalesced GPU access, and explains when that helps and when it hurts.
* :doc:`plotting` keeps plotting and report figures outside the header-only C++
  core.

The first three tutorials are ordinary C++ programs and include their expected
terminal output. The Kokkos tutorials focus on backend configuration and test
execution.

.. toctree::
   :maxdepth: 1

   basic_usage
   float_coefficients
   directional_derivatives
   cmake_package
   python_bindings
   kokkos_cpu
   kokkos_gpu
   soa_layout
   plotting
