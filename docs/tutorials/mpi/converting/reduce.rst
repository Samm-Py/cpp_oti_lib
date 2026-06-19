Global Reduction (Custom MPI_Op)
================================

The second rung: still one collective, but now it *combines* values rather than
just collecting them. We compute a global **quantity of interest** -- a single
number summed over the whole distributed field -- and because the inputs are
jets, the reduced result carries the QoI's gradient *and* Hessian with respect to
the design parameters. That is the gradient and Hessian of a global objective
over a distributed field, from one reduction, with no adjoint.

The source is ``mpi_oti_reduce/main.cpp`` at the repository root.

The Quantity Of Interest
------------------------

Each rank owns a block of an ``N × N`` grid and accumulates a partial sum of
``f(x, y; a, b) = sin(a·x)·exp(b·y)``. The design parameters ``a`` and ``b`` are
seeded as OTI variables, so the running sum is a jet:

.. code-block:: cpp

   using Jet = oti::otinum<2, 2, double>;     // value + grad + Hessian wrt (a, b)

   const Jet a = Jet::variable(0, A0);        // A0 + e_0
   const Jet b = Jet::variable(1, B0);        // B0 + e_1

   Jet local(0.0);
   for (long k = 0; k < count; ++k)
       local += oti::sin(a * x_k) * oti::exp(b * y_k);   // partial-sum jet

Every rank ends up with a *partial-sum jet*: the contribution of its block to the
global sum, together with that contribution's derivatives.

The Custom Operator
-------------------

MPI knows how to ``MPI_SUM`` an ``int`` or a ``double``, but it has no idea what
an ``otinum`` is. We teach it, by registering an operator that combines two
buffers of jets:

.. code-block:: cpp

   void jet_sum(void* in, void* inout, int* len, MPI_Datatype*) {
       const Jet* a = static_cast<const Jet*>(in);
       Jet*       b = static_cast<Jet*>(inout);
       for (int i = 0; i < *len; ++i) b[i] += a[i];      // otinum::operator+=
   }

   MPI_Op MPI_OTI_SUM;
   MPI_Op_create(&jet_sum, /*commute=*/1, &MPI_OTI_SUM);

The body is just ``otinum::operator+=`` -- OTI arithmetic plugs straight into the
reduction. ``commute=1`` says the combine is commutative (it is), letting MPI
reorder it freely. Then one collective folds every rank's partial-sum jet into
the global QoI, on every rank:

.. code-block:: cpp

   Jet global(0.0);
   MPI_Allreduce(&local, &global, 1, MPI_OTINUM, MPI_OTI_SUM, MPI_COMM_WORLD);
   const Jet qoi = global * (1.0 / TOTAL);    // mean field + its derivatives

.. note::

   For a **sum**, jet addition is coefficient-wise, so this particular reduction
   is equivalent to an ``MPI_SUM`` over the ``ncoeffs`` raw doubles of each jet --
   you could skip the custom op. The custom op is the *general* mechanism: it is
   required the moment the combine is not coefficient-wise (reducing a **product**
   of jets, for instance, where ``operator*`` is a convolution that mixes
   coefficients), and it keeps the reduction expressed in ``MPI_OTINUM`` units,
   consistent with the rest of the section.

Verification Is To A Tolerance, Not Bit-Exact
---------------------------------------------

The gather and halo examples were bit-identical to a serial run because every
element was computed identically and only moved. A reduction is different:
floating-point addition is **not associative**, so a different rank count sums in
a different order and lands on a slightly different value. Verification is
therefore tolerance-based, with two independent checks:

* **Distributed vs serial** -- the global jet against a single-process recompute,
  to a tight *relative* tolerance (``1e-10``). The gap is pure summation-order
  rounding: ``~1e-14``, and exactly ``0`` at ``np = 1``.
* **Gradient vs finite differences** -- ``d/da`` and ``d/db`` against centred
  finite differences on the parameters (``h = 1e-6``), agreeing to ``~1e-8``.
  This confirms the *reduced derivatives* are correct, independently of the OTI
  machinery.

Build And Run
-------------

.. code-block:: console

   cd mpi_oti_reduce
   mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_reduce
   mpirun -np 4 ./mpi_oti_reduce

.. code-block:: text

   ranks              : 4
   grid               : 1000 x 1000  (1000000 points)
   QoI = mean of sin(a x) exp(b y)  at a=1.000, b=1.000
     value            =  0.7898879935
     d/da             =  0.6558559384
     d/db             =  0.4598239459
     d2/da2  (Taylor) = -0.1919838760  (= QoI_aa / 2)
     d2/dadb (Taylor) =  0.3817987715  (= QoI_ab)
     d2/db2  (Taylor) =  0.1652296880  (= QoI_bb / 2)
   verify vs serial   : PASS (max relative diff 3.20e-14)
   finite difference  : centred, h = 1.0e-06
     d/da : OTI =  0.6558559384, FD =  0.6558559443, |error| = 5.96e-09
     d/db : OTI =  0.4598239459, FD =  0.4598239321, |error| = 1.38e-08
   verify gradient    : PASS (tolerance 1.0e-06)

Verified at ``np = 1, 3, 4``; the program returns nonzero if either check fails,
so it is CI-gateable. The three second-order coefficients are the QoI's Hessian
-- the curvature of a global objective over the whole distributed field -- and
they arrive in the same single reduction, at no extra communication cost.
