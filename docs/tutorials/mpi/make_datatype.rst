Committing An MPI Datatype For Jets
===================================

This tutorial builds the smallest useful MPI + OTI program and then uses it to
answer the two questions that matter before you distribute anything real: is the
answer still correct, and how does it scale. The program evaluates a function at
every point of a grid, in parallel across ranks, and gathers every jet back to
one rank. There is no halo exchange and no communication during the compute --
the only MPI traffic is one collective at the end, which is exactly where the
committed datatype earns its keep.

Nothing here depends on a GPU or on Kokkos: it is plain MPI + C++17, and the
committed datatype is the whole OTI-specific surface.

The source is ``mpi_oti_toy/`` at the repository root.

The Helper
----------

Include the optional header and ask for a datatype that describes one jet:

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

Seeding Derivatives On The Grid
-------------------------------

Each grid point seeds its coordinates as infinitesimals, so evaluating the
function returns its value *and* its derivatives at that point:

.. code-block:: cpp

   Jet x = Jet::variable(0, x0);        // x0 + e_0
   Jet y = Jet::variable(1, y0);        // y0 + e_1
   return oti::sin(x) * oti::exp(y);    // value, gradient, and Hessian

For ``otinum<2, 2, double>`` the returned jet carries the value plus all six
normalized Taylor coefficients (value, two first derivatives, three second
derivatives). MPI never looks inside this -- it only moves the bytes.

Decompose, Evaluate, Gather
---------------------------

The points are split across ranks by a flat block partition (each rank derives
its own index range from its rank number, with no communication), every rank
evaluates its slice into a local buffer, and one ``MPI_Gatherv`` assembles the
result on rank 0:

.. code-block:: cpp

   MPI_Gatherv(local.data(),  count,            MPI_OTINUM,
               global.data(), recvcounts.data(), displs.data(),
               MPI_OTINUM, 0, MPI_COMM_WORLD);

The counts and displacements are in units of ``MPI_OTINUM`` -- *jets*, not bytes.
That is the payoff of the committed datatype: MPI handles the per-element extent,
so there is no ``sizeof`` arithmetic and no chance of an off-by-a-coefficient
stride bug. (When the rank count divides the point count evenly this reduces to a
plain ``MPI_Gather``; ``Gatherv`` is used so any rank count works.)

This is also the execution model behind the whole example: a flat block
decomposition of the 1000×1000 grid, each rank evaluating the cores it is bound
to, with no communication during the compute.

Build And Run
-------------

The CPU toy is a single translation unit and needs only an MPI compiler:

.. code-block:: console

   cd mpi_oti_toy
   mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_toy
   mpirun -np 4 ./mpi_oti_toy

Rank 0 prints a sample jet and checks the gathered grid against a single-process
recompute:

.. code-block:: text

   ranks            : 4
   grid             : 1000 x 1000  (1000000 points)
   sample @ index 500500 (centre):
     value          =  0.79155923
     d/dx  (Taylor) =  1.44721739
     d/dy  (Taylor) =  0.79155923
     d2/dx2(Taylor) = -0.39577961  (= f_xx / 2)
     d2/dxdy(Taylor)=  1.44721739  (= f_xy)
     d2/dy2(Taylor) =  0.39577961  (= f_yy / 2)
   verify vs serial : PASS (bit-exact) (0 mismatching jets)

The verify line is bit-exact because every point is computed by the same
function on every rank, and MPI transfers the IEEE coefficients unchanged.
Verified at ``np = 1, 4, 7`` (7 exercises the uneven ``Gatherv`` path).

The Tightly-Packed Contract
---------------------------

``MPI_Type_contiguous(ncoeffs, MPI_DOUBLE)`` describes exactly
``ncoeffs * sizeof(double)`` contiguous bytes. That only matches the *array
stride* if the jet has no trailing padding -- i.e. if
``sizeof(Jet) == Jet::ncoeffs * sizeof(double)``. ``make_datatype`` enforces this
with a ``static_assert`` on ``oti::mpi::is_tightly_packed_v<T>``, so every
consumer gets the check for free rather than re-rolling it. The contract holds
for every ``otinum`` shape by construction; if a future layout change ever broke
it, the fix is ``MPI_Type_create_resized`` to set the extent, not removing the
assert.

The Confidence Test
-------------------

``mpi_oti_toy/test_mpi_datatype.cpp`` validates the contract directly, across the
cases the gather toy does not cover: ``float`` as well as ``double``, and
odd-``ncoeffs`` shapes (only 4- or 8-byte aligned, where any padding surprise
would surface). For each shape it asserts ``MPI_Type_size`` equals
``ncoeffs * sizeof(Coeff)``, the datatype *extent* equals ``sizeof(T)`` (the
array-stride contract behind ``count > 1`` and ``Gatherv``), and a ring
``Sendrecv`` of 257 jets round-trips bit-exact:

.. code-block:: console

   mpicxx -std=c++17 -O2 -I ../include test_mpi_datatype.cpp -o test_mpi_datatype
   mpirun -np 2 ./test_mpi_datatype

.. code-block:: text

   oti::mpi datatype confidence test (2 ranks, 257 jets/msg)
     float  <3,1> al16      ncoeffs= 4 sizeof= 16 | size OK extent OK roundtrip OK
     float  <2,2> al8       ncoeffs= 6 sizeof= 24 | size OK extent OK roundtrip OK
     float  <4,1> al4 ODD   ncoeffs= 5 sizeof= 20 | size OK extent OK roundtrip OK
     double <2,2> al16      ncoeffs= 6 sizeof= 48 | size OK extent OK roundtrip OK
     double <4,1> al8 ODD   ncoeffs= 5 sizeof= 40 | size OK extent OK roundtrip OK
   ALL PASS (0 check failures across all ranks)

The program returns nonzero on any failure, so it works as a CI gate. The odd
shapes are the meaningful rows: their extent matches ``sizeof`` exactly,
confirming the layout is packed where it is least obvious.

Derivative Accuracy
-------------------

With transport proven bit-exact, it is worth establishing what the answer *is*.
OTI returns **exact** derivatives -- it propagates the chain rule analytically,
not by finite differences -- so the only discrepancy from the closed-form
derivative is floating-point roundoff. This is a property of the OTI algebra and
the coefficient type; it has nothing to do with how many ranks you use.

``verify_derivatives`` (in ``mpi_oti_toy/``) evaluates ``f = sin(x)·exp(y)`` over
the grid for the four study algebras (``otinum<2,1>`` and ``otinum<2,2>``, each in
``float`` and ``double``), converts each jet's normalized Taylor coefficients to
partial derivatives, and compares against the analytical values
(``f_x = cos(x)e^y``, ``f_xx = -sin(x)e^y``, and so on):

.. code-block:: console

   cd mpi_oti_toy
   g++ -std=c++17 -O2 -I ../include verify_derivatives.cpp -o verify_derivatives
   ./verify_derivatives ./deriv
   python3 plot_accuracy.py deriv_errors.csv deriv_rmse.csv accuracy.png

.. image:: ../../_static/benchmarks/mpi_derivative_accuracy.png
   :alt: Box plot of OTI derivative error vs analytical, one box per algebra
   :width: 90%

Each box pools the per-point absolute errors of every derivative component for
one algebra. The result is the same story the RMSE makes precise:

.. list-table::
   :header-rows: 1

   * - Algebra
     - RMSE vs analytical
   * - ``otinum<2,1,double>``
     - 1.1e-16
   * - ``otinum<2,2,double>``
     - 1.7e-16
   * - ``otinum<2,1,float>``
     - 8.1e-08
   * - ``otinum<2,2,float>``
     - 1.1e-07

The errors sit right at each type's machine epsilon (``double`` ≈ 2.2e-16,
``float`` ≈ 1.2e-7) -- about nine orders of magnitude apart -- and many ``double``
errors are *exactly* zero. There is no truncation or step-size error to tune, as
there would be with finite differences; the only knob is the coefficient
precision. The derivative order (``<2,1>`` vs ``<2,2>``) does not change the floor,
only which derivatives are available.

Because this accuracy is a property of the **algebra**, not the parallelization,
and MPI transfers coefficients bit-for-bit (the round-trip test above confirms
this), the box plot is identical at one rank or a thousand. **Distributing the
work changes how fast you get the answer, not what the answer is** -- which is why
the rest of this page is about speed, never correctness.

Strong Scaling
--------------

``mpi_oti_scaling`` (in ``mpi_oti_toy/``) times the **compute region** -- each rank
evaluating its block of ``f(x,y) = sin(x)·exp(y)`` -- across rank counts, for all
four study algebras. This is the parallelizable work; the one-time gather is
communication and is discussed separately below, not folded into these curves.
Sweep the rank count and plot:

.. code-block:: console

   cd mpi_oti_toy
   mpicxx -std=c++17 -O2 -I ../include mpi_oti_scaling.cpp -o mpi_oti_scaling
   : > scaling.csv
   for np in 1 2 4 8 16; do
     mpirun -np $np ./mpi_oti_scaling | { [ $np -eq 1 ] && cat || tail -n +2; } >> scaling.csv
   done
   python3 plot_scaling.py scaling.csv scaling.png

.. image:: ../../_static/benchmarks/mpi_scaling.png
   :alt: MPI strong-scaling speedup and parallel efficiency vs rank count
   :width: 100%

Strong scaling (fixed problem size, more ranks) is near-linear to about 4 ranks,
then tapers -- a speedup of roughly 6× at 16 ranks on this 8-core laptop (the
efficiency curve falls from ~95% at 2 ranks to ~40% at 16). Two honest takeaways:

* **The taper is expected.** As ranks rise the per-rank slice shrinks (at 16
  ranks each rank holds only ~62,500 points), so fixed per-iteration overhead and
  memory bandwidth start to dominate, and on a laptop the cores beyond the
  physical count (hyperthreads) add little. This is ordinary strong-scaling /
  Amdahl behavior, not an OTI effect.
* **Heavier jets amortize overhead.** The larger ``<2,2>`` algebras do slightly
  more arithmetic per point, so they hold efficiency marginally better at high
  rank counts than the lightest ``<2,1,float>`` -- more compute per unit of
  overhead.

The Gather Is Communication
---------------------------

The final ``MPI_Gatherv`` is separate from the scaled compute: it moves the whole
result to one rank (for ``<2,2,double>``, 1,000,000 jets × 48 bytes ≈ 48 MB). On a
real solver you would keep data distributed and exchange only what neighbors need
rather than gathering everything -- which is exactly the progression in
:doc:`converting/index`.
