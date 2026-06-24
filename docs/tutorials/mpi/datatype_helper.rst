The Datatype Helper
===================

The optional header ``otinum/mpi.hpp`` provides the committed jet datatype through
``oti::mpi::make_datatype<T>()``. It is deliberately **not** part of the
``otinum.hpp`` umbrella, so the core headers stay MPI-free and non-MPI builds
carry no dependency. Include it and ask for a datatype that describes one jet:

.. code-block:: cpp

   #include "otinum/otinum.hpp"
   #include "otinum/mpi.hpp"

   using Jet = oti::otinum<2, 2, double>;   // value + grad + Hessian, 6 coeffs

   MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
   // ... use MPI_OTINUM in sends / receives / collectives ...
   oti::mpi::free_datatype(MPI_OTINUM);

``make_datatype<T>()`` builds and commits ``MPI_Type_contiguous(T::ncoeffs,
MPI_DOUBLE)`` (or ``MPI_FLOAT`` for a ``float`` coefficient type) and returns the
committed handle. The caller owns the handle and releases it with
``free_datatype`` (a thin wrapper over ``MPI_Type_free``). Commit it once per
``(M, N, Coeff)`` shape at setup and reuse it for every message; it is also the
base element for the derived datatypes a communicating solver needs later
(``MPI_Type_vector`` for strided halos, ``MPI_Type_indexed`` for unstructured
ghost-node lists).

Nothing about this depends on a GPU or on Kokkos -- the datatype describes a jet
in plain host memory, and the same committed handle works unchanged for
device-resident buffers if you do bring Kokkos.
