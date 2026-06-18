Converting Code To OTI
======================

These tutorials take an existing ``double`` MPI (and Kokkos) program and
OTI-enable it, so it produces derivatives. Each is a concrete **before/after**
pair, and they are ordered by how much the ranks have to communicate -- the only
thing that really changes as the problems get harder. The OTI side of each step
is small; the new work is the communication pattern.

The ladder
----------

#. **Independent evaluation (gather)** -- no communication during the compute;
   each rank evaluates its block and the results are gathered. The OTI change is
   just the scalar type, the seeding, and the MPI datatype. *(:doc:`gather`)*
#. **Global reduction** *(upcoming)* -- reduce a quantity of interest across ranks
   to a global sensitivity, which needs a custom ``MPI_Op`` that sums jets.
#. **Halo exchange** *(upcoming)* -- nearest-neighbor communication for a
   structured stencil, built from ``MPI_Type_vector`` over the jet datatype.
#. **Unstructured meshes** *(upcoming)* -- arbitrary ghost-node lists via
   ``MPI_Type_indexed``.

The committed jet datatype from :doc:`../make_datatype` is the common building
block throughout; later rungs wrap derived datatypes and reduction operators
around it. Start with the first:

.. toctree::
   :maxdepth: 1

   gather
