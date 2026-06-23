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

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Converting Code to OTI

   converting/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Correctness & Scaling

   verification

The Datatype Helper
-------------------

The optional header ``otinum/mpi.hpp`` provides that datatype through
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

Message Volume
--------------

Counting in *jets* keeps the API clean, but do not lose sight of what one jet
costs on the wire: a jet is ``ncoeffs`` coefficients, so **every message is**
``ncoeffs`` **times the size of the equivalent scalar message**. The coefficient
count grows combinatorially with the number of seeded directions :math:`M` and the
derivative order :math:`N`, as :math:`\binom{M+N}{N}`:

.. list-table::
   :header-rows: 1

   * - Algebra
     - Carries
     - ``ncoeffs``
     - Bytes (``double``) vs scalar
   * - ``otinum<2,1>``
     - value + gradient
     - 3
     - 24 B (3×)
   * - ``otinum<2,2>``
     - value + gradient + Hessian
     - 6
     - 48 B (6×)
   * - ``otinum<3,2>``
     - 3 directions, 2nd order
     - 10
     - 80 B (10×)
   * - ``otinum<3,3>``
     - 3 directions, 3rd order
     - 20
     - 160 B (20×)

This is a multiplier on **communication**, not just storage: a halo exchange, a
gather, or an ``Allreduce`` moves ``ncoeffs`` times the bytes a plain-``double``
solve would, and bandwidth-bound exchanges scale accordingly. The practical rule
is to **seed only the directions you need and use the lowest derivative order that
answers the question** -- the jet shape, not the rank count, sets the volume. The
:doc:`verification` page measures the compute side of the same trade-off.

This is the whole OTI-specific surface for MPI.
:doc:`Converting Code to OTI <converting/index>` is a growing ladder of
before/after examples, ordered by communication complexity, that put it to work,
and :doc:`Correctness & Scaling <verification>` validates the datatype and the OTI
derivatives and measures how the evaluation scales. The separate
:doc:`../integration` tutorial is the broader culmination -- bringing this
together with Kokkos for a full MPI + GPU application.

The example sources live at the repository root in ``mpi_oti_convert/``
(conversion before/after, the first rung), ``mpi_oti_reduce/`` (global reduction
with a custom ``MPI_Op``), ``mpi_oti_halo/`` (the Jacobi halo-exchange solver),
``mpi_oti_unstructured/`` (irregular ghost lists via ``MPI_Type_indexed``), and
``mpi_oti_toy/`` (the datatype/accuracy/scaling verification harness); the
optional GPU sources in ``mpi_oti_gpu_toy/`` are covered by the
:doc:`../integration` tutorial.
