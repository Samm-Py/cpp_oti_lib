Correctness And Scaling
=======================

The MPI examples move jets correctly by construction, but before distributing
anything real, two questions matter: is the answer still correct, and how does it
scale. A small companion harness in ``mpi_oti_toy/`` answers both. It is
deliberately independent of any one kernel -- it sweeps datatype **shapes** and
**algebras** -- so it validates the datatype and the OTI derivatives in general,
not just one example. The conversion rungs (:doc:`converting/gather`,
:doc:`converting/reduce`, :doc:`converting/halo`) each carry their own bit-exact
check; this page is the shared foundation they rest on.

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
cases a single example does not cover: ``float`` as well as ``double``, and
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
a grid for the four study algebras (``otinum<2,1>`` and ``otinum<2,2>``, each in
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
work changes how fast you get the answer, not what the answer is.**

Strong Scaling
--------------

``mpi_oti_scaling`` (in ``mpi_oti_toy/``) times the **compute region** -- each rank
evaluating its block of ``f(x,y) = sin(x)·exp(y)`` -- across rank counts, for all
four study algebras. This is the parallelizable work; the one-time gather is
communication and is not folded into these curves. Sweep the rank count and plot:

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

* **The taper is expected.** As ranks rise the per-rank slice shrinks, so fixed
  per-iteration overhead and memory bandwidth start to dominate, and on a laptop
  the cores beyond the physical count (hyperthreads) add little. This is ordinary
  strong-scaling / Amdahl behavior, not an OTI effect.
* **Heavier jets amortize overhead.** The larger ``<2,2>`` algebras do slightly
  more arithmetic per point, so they hold efficiency marginally better at high
  rank counts than the lightest ``<2,1,float>`` -- more compute per unit of
  overhead.

A one-time gather of the whole field (for ``<2,2,double>``, 1,000,000 jets × 48
bytes ≈ 48 MB) is the price of collecting everything on one rank. A real solver
keeps data distributed and exchanges only what neighbors need -- which is exactly
the progression through the ladder (:doc:`converting/reduce`,
:doc:`converting/halo`).
