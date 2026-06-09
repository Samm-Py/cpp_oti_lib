Tutorials
=========

These tutorials focus on practical workflows:

* :doc:`basic_usage` verifies first- and second-order derivatives against
  analytic values.
* :doc:`float_coefficients` shows how to use ``float`` storage instead of the
  default ``double`` storage.
* :doc:`directional_derivatives` shows how to seed a line through a larger
  physical input space and compute derivatives along that direction.
* :doc:`kokkos_cpu` and :doc:`kokkos_gpu` build the Kokkos smoke test on CPU
  and CUDA backends.
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
   kokkos_cpu
   kokkos_gpu
   plotting
