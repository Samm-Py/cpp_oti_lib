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
* :doc:`scaling_and_accuracy` measures that same evaluation: how it speeds up
  with more ranks, and a box plot showing OTI derivatives match the analytical
  values to each algebra's floating-point precision floor.
* :doc:`gpu` builds the one-rank-per-GPU execution model -- binding each rank to
  a device and giving it an exclusive turn via a token ring -- and runs it on a
  single GPU by simulating the multi-GPU case.
* :doc:`integration` is the culmination: how to bring ``cpp_oti_lib`` into your
  own MPI + Kokkos application -- the dependency model, the CMake recipe, the
  device-pointer vs host-staging transport choice, and the toolchain gotchas that
  bite when you stack the three together.

The example sources live at the repository root in ``mpi_oti_toy/`` (CPU) and
``mpi_oti_gpu_toy/`` (GPU).

.. toctree::
   :maxdepth: 1

   make_datatype
   scaling_and_accuracy
   gpu
   integration
