Halo Exchange (Jacobi Stencil)
==============================

The third rung, and the first solver that **communicates during the compute**.
The gather example (:doc:`gather`) had no traffic until the very end; here the
ranks exchange a ghost layer with their neighbours on *every* iteration. That
exchange is the new work -- the OTI side is, as always, just the scalar type and
seeding the inputs.

The problem is steady-state heat on the unit square: solve Laplace's equation
with a 5-point Jacobi stencil,

.. math::

   u^{t+1}_{i,j} = \tfrac{1}{4}\,\bigl(u^{t}_{i-1,j} + u^{t}_{i+1,j}
                                       + u^{t}_{i,j-1} + u^{t}_{i,j+1}\bigr),

distributed over a 2D Cartesian grid of ranks. The source is
``mpi_oti_halo/main.cpp`` at the repository root.

The OTI Angle: Sensitivities For Free
-------------------------------------

The two hot walls carry their temperature as **seeded variables** rather than
plain numbers:

.. code-block:: cpp

   using Jet = oti::otinum<2, 1, double>;          // value + d/dT_west + d/dT_south

   const Jet T_WEST  = Jet::variable(0, 1.0);      // 1.0 + e_0
   const Jet T_SOUTH = Jet::variable(1, 1.0);      // 1.0 + e_1
   const Jet T_COLD  = Jet(0.0);

Every cell then converges to a jet whose coefficients are the temperature *and*
its derivative with respect to each wall -- the parameter sensitivity of the
entire field, from one solve. The stencil code is unchanged; the arithmetic is
overloaded, so ``(a + b + c + d) * 0.25`` propagates the derivatives
automatically.

Decomposition: A 2D Cartesian Grid
----------------------------------

Letting MPI choose the process grid and wiring up neighbours is a few lines:

.. code-block:: cpp

   int dims[2] = {0, 0};
   MPI_Dims_create(world_size, 2, dims);          // factor ranks into Px x Py
   int periods[2] = {0, 0};                        // non-periodic: walls, not torus
   MPI_Comm cart;
   MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1, &cart);

   int up, down, left, right;
   MPI_Cart_shift(cart, 0, 1, &up, &down);         // dim 0 = rows (i)
   MPI_Cart_shift(cart, 1, 1, &left, &right);      // dim 1 = columns (j)

Each rank owns an ``nx × ny`` interior tile wrapped in a one-cell **ghost layer**
(local array stride ``ny + 2``). ``MPI_Cart_shift`` returns ``MPI_PROC_NULL`` for
neighbours off the edge of the grid, which is the key simplification for the
boundary: ghost layers that coincide with a Dirichlet wall are filled once with
the wall temperature and never touched again, because a ``Sendrecv`` to or from
``MPI_PROC_NULL`` is a no-op.

The Two Halo Datatypes
----------------------

This is where the committed jet element pays off a second time. Both halos are
described in units of *jets*, built on ``MPI_OTINUM``:

.. code-block:: cpp

   MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();

   // Column halo (E/W): nx jets, one per row, strided by `stride` jets.
   MPI_Datatype MPI_COL;
   MPI_Type_vector(nx, 1, stride, MPI_OTINUM, &MPI_COL);
   MPI_Type_commit(&MPI_COL);
   // Row halo (N/S) is contiguous -> just `ny` of MPI_OTINUM, no derived type.

A **row** (north/south neighbour) is a contiguous run of ``ny`` jets, so it ships
as ``count = ny`` of ``MPI_OTINUM`` with no extra type. A **column** (east/west
neighbour) is one jet per row, separated by a full row stride, so it needs
``MPI_Type_vector`` *over* the jet element -- a derived datatype on a derived
datatype. The stride is in jets, not bytes: MPI knows the extent of one
``MPI_OTINUM`` is exactly ``sizeof(Jet)`` (the tightly-packed contract from
:doc:`../make_datatype`), so the strided gather lands on jet boundaries with no
byte arithmetic.

The Exchange
------------

Each iteration, before sweeping the interior, every rank swaps edges with its
neighbours using ``MPI_Sendrecv`` (which avoids deadlock without manual ordering):

.. code-block:: cpp

   // Rows (contiguous, count = ny): send an edge interior row, receive a ghost row.
   MPI_Sendrecv(&cur[idx(nx, 1)], ny, MPI_OTINUM, down, 0,
                &cur[idx(0, 1)],  ny, MPI_OTINUM, up,   0, cart, MPI_STATUS_IGNORE);
   MPI_Sendrecv(&cur[idx(1, 1)],      ny, MPI_OTINUM, up,   1,
                &cur[idx(nx + 1, 1)], ny, MPI_OTINUM, down, 1, cart, MPI_STATUS_IGNORE);

   // Columns (strided, one MPI_COL): same pattern in the j direction.
   MPI_Sendrecv(&cur[idx(1, ny)], 1, MPI_COL, right, 2,
                &cur[idx(1, 0)],  1, MPI_COL, left,  2, cart, MPI_STATUS_IGNORE);
   MPI_Sendrecv(&cur[idx(1, 1)],      1, MPI_COL, left,  3,
                &cur[idx(1, ny + 1)], 1, MPI_COL, right, 3, cart, MPI_STATUS_IGNORE);

   jacobi_sweep(cur, next, nx, ny, stride);
   std::swap(cur, next);

Edge ranks send to ``MPI_PROC_NULL`` on their outward sides, so those calls do
nothing and the Dirichlet ghosts survive.

Verify Without A Gather
-----------------------

Jacobi at iteration *t* depends only on the full field at *t-1*, so it is
fully deterministic regardless of how the domain is split. The example exploits
that: every rank redundantly runs the *identical* serial solve on the whole
domain (the stencil function ``jacobi_sweep`` is shared verbatim by both paths)
and compares its own tile against the matching subblock. The distributed result
is **bit-for-bit identical** to serial -- including every derivative coefficient
-- because the halo exchange transports the exact IEEE bytes and the arithmetic
order never changes. No ``MPI_Gatherv`` is needed; the only real communication is
the halo exchange itself, which is the point of this rung.

Build And Run
-------------

.. code-block:: console

   cd mpi_oti_halo
   mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_halo
   mpirun -np 4 ./mpi_oti_halo

.. code-block:: text

   sample @ global centre (64,64), owned by cart rank 0 (0,0):
     temperature        =  0.25964034
     d/dT_west          =  0.12982017
     d/dT_south         =  0.12982017
   ---
   process grid       : 2 x 2  (4 ranks)
   interior grid      : 128 x 128  (4000 Jacobi iterations)
   verify vs serial   : PASS (bit-exact) (0 mismatching jets)

``MPI_Dims_create`` factors the rank count into the process grid: ``4 → 2×2``,
``6 → 3×2``, ``9 → 3×3`` (these exercise the strided column halo), while a prime
count gives a 1D strip. Verified bit-exact at ``np = 1, 2, 3, 4, 6, 7, 9``; the
program returns nonzero on any mismatch, so it is CI-gateable.

The sample output doubles as a correctness check on the *derivatives*: the
problem is linear and both walls are at 1.0, so the temperature equals the sum of
the two sensitivities, and by symmetry across the diagonal the West and South
sensitivities are equal -- which is exactly what the jet reports.
