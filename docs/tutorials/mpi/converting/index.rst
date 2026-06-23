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

    // 4. describe the jet to MPI: ONE committed datatype, the base for EVERY op
   -    MPI_Datatype JET = MPI_DOUBLE;
   +    MPI_Datatype JET = oti::mpi::make_datatype<Scalar>();
        //   movement (scatter / gather / bcast) : pass JET directly, counts in jets
        //   structured / unstructured halos     : MPI_Type_vector / _indexed over JET
        //   reductions (reduce / allreduce)     : pair JET with a custom MPI_Op
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
   counts are then in *jets*, not bytes. This one committed type is the base for
   **every** MPI operation -- passed directly to movement collectives, wrapped in a
   derived datatype (``MPI_Type_vector`` / ``MPI_Type_indexed``) for halos, or
   paired with a custom ``MPI_Op`` for reductions -- and it is freed at the end.
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

Edit 4 above is the **only** edit that varies between programs, and it varies in a
small, bounded way: each communication pattern adds exactly **one** piece of MPI
built on the committed jet datatype ``JET``. The patterns are ordered by how much
the ranks must communicate -- the real axis of difficulty:

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Rung
     - New MPI surface (on top of the jet datatype)
     - Full example
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

The rule of thumb: **movement** collectives (scatter / gather / broadcast /
allgather) need *only* the datatype; **combining** collectives (reduce) also need a
custom operator; **communicating solvers** also wrap a derived datatype around the
jet. Each pattern's exact addition is below; the linked example is the full
before/after walkthrough.

**Independent evaluation -- movement only.** Scatter the field, transform each
block locally, gather the results. Movement collectives move jets directly under
the committed datatype -- **no custom operator** -- so the result is bit-exact and
identical at any rank count:

.. code-block:: diff

   -    MPI_Datatype JET = MPI_DOUBLE;
   +    MPI_Datatype JET = oti::mpi::make_datatype<Scalar>();   // the only new type

        // movement collectives are unchanged -- they just carry JET
        MPI_Bcast(&param, 1, JET, 0, comm);
        MPI_Scatterv(in.data(),  counts, displs, JET,
                     local.data(), my_n,         JET, 0, comm);
        MPI_Gatherv (local.data(), my_n,         JET,
                     out.data(), counts, displs, JET, 0, comm);
   +    oti::mpi::free_datatype(JET);

Full walkthrough: :doc:`gather`.

**Global reduction -- a custom operator.** Combining collectives must know how to
*fold* two jets, which MPI cannot infer, so they take a user ``MPI_Op``. The header
ships ``make_sum_op`` / ``make_prod_op`` / ``make_maxloc_op`` / ``make_minloc_op``
(or pass your own combine). The gradient and Hessian reduce in the same collective
as the value:

.. code-block:: diff

   -    MPI_Allreduce(local.data(), global.data(), n, MPI_DOUBLE, MPI_SUM, comm);
   +    MPI_Op SUM = oti::mpi::make_sum_op<Scalar>();   // fold jets coefficient-wise
   +    MPI_Allreduce(local.data(), global.data(), n, JET, SUM, comm);
   +    oti::mpi::free_op(SUM);

Full walkthrough: :doc:`reduce`.

**Halo exchange -- a strided derived datatype.** A structured stencil exchanges
edge layers every iteration. Contiguous edges (a row) ship as a plain ``count`` of
``JET``; strided edges (a column) need one ``MPI_Type_vector`` *over* the jet -- a
derived datatype on a derived datatype, in units of jets:

.. code-block:: diff

        MPI_Datatype COL;
   -    MPI_Type_vector(nx, 1, stride, MPI_DOUBLE, &COL);   // a column of doubles
   +    MPI_Type_vector(nx, 1, stride, JET,        &COL);   // a column of jets
        MPI_Type_commit(&COL);

        // a row is contiguous (count = ny of JET); a column uses COL
        MPI_Sendrecv(&u[col_send], 1, COL, left,  0,
                     &u[col_recv], 1, COL, right, 0, comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&COL);

Full walkthrough: :doc:`halo`.

**Unstructured meshes -- an indexed derived datatype.** An unstructured mesh has no
grid regularity: the nodes a neighbour needs are an arbitrary, scattered subset of
this rank's owned nodes. ``MPI_Type_indexed`` describes that subset by its
displacements (in jets) and gathers it in place, with no manual packing; the
receiver lands it in a contiguous ghost block:

.. code-block:: diff

        // displ = the scattered owned-node slots this neighbour needs (sorted by id)
        std::vector<int> blocklen(displ.size(), 1);
        MPI_Datatype SEND;
   -    MPI_Type_indexed(displ.size(), blocklen.data(), displ.data(), MPI_DOUBLE, &SEND);
   +    MPI_Type_indexed(displ.size(), blocklen.data(), displ.data(), JET,        &SEND);
        MPI_Type_commit(&SEND);

        MPI_Sendrecv(buf, 1, SEND, nbr, 0,                        // scattered send
                     &buf[ghost_start], ghost_count, JET, nbr, 0, // contiguous recv
                     comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&SEND);

Full walkthrough: :doc:`unstructured`.

Everything else is shared: the committed jet datatype from :doc:`../index` is the
common building block under all of these.

Examples
--------

The full before/after walkthroughs for each pattern above -- each a concrete
``double`` → OTI conversion of a small, self-contained MPI program that verifies
its own results:

* :doc:`gather` -- independent evaluation (movement only).
* :doc:`reduce` -- global reduction with a custom ``MPI_Op``.
* :doc:`halo` -- structured halo exchange (``MPI_Type_vector``).
* :doc:`unstructured` -- unstructured ghost lists (``MPI_Type_indexed``).

.. toctree::
   :hidden:
   :maxdepth: 1

   gather
   reduce
   halo
   unstructured
