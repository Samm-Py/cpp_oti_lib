Committing An MPI Datatype For Jets
===================================

This tutorial builds the smallest useful MPI + OTI program: evaluate a function
at every point of a grid, in parallel across ranks, and gather every jet back to
one rank. There is no halo exchange and no communication during the compute --
the only MPI traffic is one collective at the end, which is exactly where the
committed datatype earns its keep.

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
