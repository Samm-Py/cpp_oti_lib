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
  multi-index access, adding new instantiations, and plotting with
  Matplotlib.
* :doc:`kokkos_cpu` and :doc:`kokkos_gpu` build the Kokkos smoke test on CPU
  and CUDA backends.
* :doc:`soa_layout` stores arrays of OTI numbers coefficient-major for
  coalesced GPU access, introduces the library's conditional object alignment,
  and explains when each helps and when it hurts.
* :doc:`mpi/index` distributes OTI numbers across MPI ranks: a committed
  ``MPI_Datatype`` for one jet, an embarrassingly-parallel gather, a datatype
  confidence test, and the same gather for GPU-resident jets.

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
   mpi/index
