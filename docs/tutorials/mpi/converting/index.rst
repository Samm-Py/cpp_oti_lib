Converting Code To OTI
======================

These tutorials take an existing ``double`` MPI (and Kokkos) program and
OTI-enable it, so it produces derivatives. Each is a concrete **before/after**
pair, and they are ordered by how much the ranks have to communicate -- the only
thing that really changes as the problems get harder. The OTI side of each step
is small; the new work is the communication pattern.

.. note::

   **The conversion does not require Kokkos.** The OTI-specific changes -- the
   scalar type swap, seeding the inputs as variables, the MPI datatype, and
   reading the derivatives out -- are identical whether the code is serial,
   OpenMP, MPI-only, or Kokkos. Kokkos adds exactly **one** thing: define
   ``OTI_ENABLE_KOKKOS`` so a jet is callable inside a device (GPU) kernel.
   Without it, ``otinum`` is a plain header-only value type that works in ordinary
   CPU code unchanged, and ``oti::mpi::make_datatype`` is unaffected either way
   (the jet's layout is the same with ``std::array`` or ``Kokkos::Array``). The
   examples here use Kokkos because that is the common target, but a non-Kokkos
   MPI program converts line-for-line the same -- just drop the ``View``, the
   ``KOKKOS_LAMBDA``, and that one flag.

The ladder
----------

#. **Independent evaluation (gather)** -- no communication during the compute;
   each rank evaluates its block and the results are gathered. The OTI change is
   just the scalar type, the seeding, and the MPI datatype. *(:doc:`gather`)*
#. **Global reduction** *(upcoming)* -- reduce a quantity of interest across ranks
   to a global sensitivity, which needs a custom ``MPI_Op`` that sums jets.
#. **Halo exchange** -- nearest-neighbor communication for a structured Jacobi
   stencil, built from ``MPI_Type_vector`` over the jet datatype. The first solver
   that communicates every iteration. *(:doc:`halo`)*
#. **Unstructured meshes** *(upcoming)* -- arbitrary ghost-node lists via
   ``MPI_Type_indexed``.

The committed jet datatype from :doc:`../make_datatype` is the common building
block throughout; later rungs wrap derived datatypes and reduction operators
around it. Start with the first:

.. toctree::
   :maxdepth: 1

   gather
   halo
