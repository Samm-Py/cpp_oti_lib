Converting Code To OTI
======================

These tutorials take an existing ``double`` MPI program and OTI-enable it, so it
produces derivatives. Each is a concrete **before/after** pair, and they are
ordered by how much the ranks have to communicate -- the only thing that really
changes as the problems get harder. The OTI side of each step is small; the new
work is the communication pattern.

.. note::

   **The examples here are plain MPI -- no Kokkos, no GPU.** The OTI-specific
   changes -- the scalar type swap, seeding the inputs as variables, the MPI
   datatype, and reading the derivatives out -- are identical whether the code is
   serial, OpenMP, MPI-only, or Kokkos, so the simplest setting shows them most
   clearly. In an application already configured for Kokkos, the additional
   **OTI-specific** setting is ``OTI_ENABLE_KOKKOS``, which makes a jet callable
   inside a device kernel. That switches the coefficient container from
   ``std::array`` to ``Kokkos::Array`` but does not change the jet's layout, so
   ``oti::mpi::make_datatype`` is unaffected. The Kokkos backend, compiler
   wrapper, GPU architecture, and MPI transport setup are covered in
   :doc:`../integration`.

The tutorials are ordered as a ladder by communication complexity:

#. **Independent evaluation (scatter / gather)** -- no communication during the
   compute; the root scatters the field, each rank transforms its block, and the
   results are gathered. The OTI change is just the scalar type, the seeding, and
   the MPI datatype. *(:doc:`gather`)*
#. **Global reduction** -- reduce a quantity of interest across ranks to a global
   gradient and Hessian, using a custom ``MPI_Op`` that sums jets. *(:doc:`reduce`)*
#. **Halo exchange** -- nearest-neighbor communication for a structured Jacobi
   stencil, built from ``MPI_Type_vector`` over the jet datatype. The first solver
   that communicates every iteration. *(:doc:`halo`)*
#. **Unstructured meshes** -- arbitrary ghost-node lists via ``MPI_Type_indexed``,
   for a graph with no grid regularity. The send set is a scattered subset of each
   rank's owned nodes. *(:doc:`unstructured`)*

The committed jet datatype from :doc:`../index` is the common building
block throughout; later rungs wrap derived datatypes and reduction operators
around it. Start with the first:

.. toctree::
   :maxdepth: 1

   gather
   reduce
   halo
   unstructured
