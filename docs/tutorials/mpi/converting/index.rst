Converting Code To OTI
======================

These tutorials take an existing ``double`` MPI program and OTI-enable it, so it
produces derivatives. The OTI-specific edits are **the same few lines every
time**, no matter how the ranks communicate; what changes from one problem to the
next is only the communication pattern. This page shows the common edits once,
then routes you to a worked before/after example for each communication pattern.

What Actually Changes
---------------------

Whatever the MPI program does -- scatter/gather, reduce, halo exchange,
unstructured ghost lists -- OTI-enabling it is the same five edits. The kernel and
the communication *structure* do not change; the overloaded arithmetic carries the
derivatives through unmodified code.

.. code-block:: diff

    // 1. include the optional headers (value type + math, and the MPI helper)
   +#include "otinum/otinum.hpp"
   +#include "otinum/mpi.hpp"

    // 2. swap the scalar type the kernel computes in -- one alias
   -using Scalar = double;
   +using Scalar = oti::otinum<2, 1, double>;   // value + 2 derivative directions

    // 3. seed the inputs you want derivatives with respect to
   -    Scalar x = x0;
   +    Scalar x = Scalar::variable(0, x0);     // x0 + unit perturbation in direction 0

    // 4. describe the jet to MPI: ONE committed datatype replaces MPI_DOUBLE
   -    MPI_Datatype JET = MPI_DOUBLE;
   +    MPI_Datatype JET = oti::mpi::make_datatype<Scalar>();
        // ... use JET in every send / recv / collective, in units of jets ...
   +    oti::mpi::free_datatype(JET);

    // 0. THE KERNEL AND THE COMMUNICATION STRUCTURE ARE UNCHANGED
    //    overloaded operators + <cmath> propagate the derivatives automatically

    // 5. read the derivatives out of the result
   +    double dfdx = result.coeff(oti::sparse({{0, 1}}));   // d(result)/dx

What each edit is for:

#. **Include the optional headers.** ``otinum/otinum.hpp`` brings the value type
   and the ``oti::`` math overloads; ``otinum/mpi.hpp`` adds the datatype helper.
   Neither is in the core umbrella, so non-OTI builds are unaffected.
#. **Change the scalar type.** One ``using`` alias. Everything typed on ``Scalar``
   -- buffers, the kernel -- follows automatically. ``<M, N>`` chooses how many
   directions and which derivative order; see :doc:`../index` for the cost.
#. **Seed the inputs as variables.** The one genuinely new line of *logic*:
   declaring which quantities you want derivatives with respect to.
   ``Scalar::variable(d, x0)`` is ``x0`` carrying a unit infinitesimal in
   direction ``d``.
#. **Describe the element to MPI.** ``oti::mpi::make_datatype<Scalar>()`` commits
   one contiguous datatype for the jet and replaces ``MPI_DOUBLE`` everywhere;
   counts are then in *jets*, not bytes. This single committed type is also the
   base element for the derived datatypes and reduction operators the harder rungs
   need (below), and it is freed at the end.
#. **Read the derivatives out.** The result is now a jet; pull a coefficient with
   ``coeff(oti::sparse(...))`` (or feed it straight into more OTI arithmetic).

That is the entire OTI surface. Note what is **absent**: the kernel, the domain
decomposition, and the collective calls keep their exact shape.

.. note::

   **The examples are plain MPI -- no Kokkos, no GPU.** These five edits are
   identical whether the code is serial, OpenMP, MPI-only, or Kokkos, so the
   simplest setting shows them most clearly. In an application already configured
   for Kokkos, the one extra **OTI-specific** setting is ``OTI_ENABLE_KOKKOS``,
   which makes a jet callable inside a device kernel (it switches the coefficient
   container from ``std::array`` to ``Kokkos::Array`` but does not change the
   layout, so ``oti::mpi::make_datatype`` is unaffected). The Kokkos backend,
   compiler wrapper, GPU architecture, and MPI transport setup are covered in
   :doc:`../../integration`.

What Changes Per Example: The Communication Pattern
---------------------------------------------------

Edit 4 above is where the rungs diverge. Each example is a concrete before/after
pair; they are ordered by how much the ranks must communicate -- the real axis of
difficulty -- and each adds exactly **one** new piece of MPI built on the committed
jet datatype:

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Rung
     - New MPI surface (on top of the jet datatype)
     - Example
   * - **Independent evaluation**
     - just the datatype -- ``Scatterv`` / ``Bcast`` / ``Gatherv`` move jets
       directly (no custom op)
     - :doc:`gather`
   * - **Global reduction**
     - a custom ``MPI_Op`` (``oti::mpi::make_sum_op``) to fold jets in
       ``Reduce`` / ``Allreduce``
     - :doc:`reduce`
   * - **Halo exchange**
     - ``MPI_Type_vector`` over the jet for strided structured halos; first solver
       that communicates every iteration
     - :doc:`halo`
   * - **Unstructured meshes**
     - ``MPI_Type_indexed`` over the jet for arbitrary, scattered ghost-node lists
     - :doc:`unstructured`

Movement collectives (scatter / gather / broadcast / allgather) need *only* the
datatype; combining collectives (reduce) additionally need a custom operator;
communicating solvers additionally wrap a derived datatype around the jet. The
committed jet datatype from :doc:`../index` is the common building block
throughout, and :doc:`../verification` validates it and the derivatives across
shapes and rank counts.

.. toctree::
   :maxdepth: 1

   gather
   reduce
   halo
   unstructured
