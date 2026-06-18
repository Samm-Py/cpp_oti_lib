MPI
===

These tutorials distribute OTI numbers across MPI ranks. They rely on a single
fact about the layout: an ``oti::otinum<M, N, Coeff>`` is a fixed-size,
contiguous block of ``ncoeffs`` coefficients with no pointers, no heap, and no
trailing padding (see :doc:`../soa_layout` for the alignment rule that keeps it
padding-free). An array of jets is therefore one packed buffer, and a single
committed ``MPI_Datatype`` describes one jet -- so MPI moves jets as a
first-class element, with send/receive/collective counts expressed in *jets*
rather than bytes. There is no serialization layer to write.

The optional header ``otinum/mpi.hpp`` provides that datatype through
``oti::mpi::make_datatype<T>()``. It is deliberately **not** part of the
``otinum.hpp`` umbrella, so the core headers stay MPI-free and non-MPI builds
carry no dependency.

* :doc:`make_datatype` introduces ``otinum/mpi.hpp`` and an
  embarrassingly-parallel grid evaluation that gathers every jet to one rank,
  then verifies the datatype layout with a focused confidence test.
* :doc:`gpu` runs the same gather for jets produced on a CUDA device, choosing
  the device-pointer or host-staging transport at runtime, and shows why the
  datatype itself is GPU-agnostic.
* :doc:`integration` is the culmination: how to bring ``cpp_oti_lib`` into your
  own MPI + Kokkos application -- the dependency model, the CMake recipe, and the
  toolchain gotchas that bite when you stack the three together.

The example sources live at the repository root in ``mpi_oti_toy/`` (CPU) and
``mpi_oti_gpu_toy/`` (GPU).

.. toctree::
   :maxdepth: 1

   make_datatype
   gpu
   integration
