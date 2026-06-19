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

Nothing about this depends on a GPU or on Kokkos -- the datatype describes a jet
in plain host memory, and the same committed handle works unchanged for
device-resident buffers if you do bring Kokkos. **Working Example** is the
hardware-independent foundation: a single MPI + C++17 program that commits the
datatype, distributes an evaluation, gathers it back, and then measures both its
accuracy and its scaling. **Converting Code to OTI** is a growing ladder of
before/after examples, ordered by communication complexity. **Reference** is the
full integration guide for stacking OTI with MPI and Kokkos together.

The example sources live at the repository root in ``mpi_oti_toy/`` (CPU),
``mpi_oti_convert/`` (conversion before/after), and ``mpi_oti_halo/`` (the Jacobi
halo-exchange solver); the optional GPU sources in ``mpi_oti_gpu_toy/`` are
covered by the integration reference.

.. toctree::
   :maxdepth: 1
   :caption: Working Example

   make_datatype

.. toctree::
   :maxdepth: 1
   :caption: Converting Code to OTI

   converting/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   integration
