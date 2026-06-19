Global Reduction (Custom MPI_Op)
================================

The second rung: still one collective, but now it *combines* values rather than
just collecting them. We compute a global **quantity of interest** -- a single
number summed over the whole distributed field -- and because the inputs are
jets, the reduced result carries the QoI's gradient *and* Hessian with respect to
the design parameters. That is the gradient and Hessian of a global objective
over a distributed field, from one reduction, with no adjoint.

The before/after sources are ``mpi_oti_reduce/main_before.cpp`` (plain ``double``)
and ``main.cpp`` (OTI); the differences are the changes below.

The Starting Point
------------------

``main_before.cpp`` is an ordinary MPI reduction: each rank sums its block of
``f(x, y; a, b) = sin(a·x)·exp(b·y)`` over an ``N × N`` grid, and one
``MPI_Allreduce(MPI_SUM)`` over ``double`` produces the global mean. It computes
the QoI *value* and nothing else.

The Changes
-----------

.. code-block:: diff

   -using Scalar = double;
   +#include "otinum/otinum.hpp"                    // 1. otinum core
   +#include "otinum/mpi.hpp"                        //    datatype + reduction op
   +using Jet = oti::otinum<2, 2, double>;           // 2. value + grad + Hessian

    // design parameters
   -const Scalar a = A0;
   -const Scalar b = B0;
   +const Jet a = Jet::variable(0, A0);              // 3. seed a = A0 + e_0
   +const Jet b = Jet::variable(1, B0);              //    seed b = B0 + e_1

    // partial sum over this rank's block -- THE LOOP DOES NOT CHANGE
   -Scalar local = 0.0;
   +Jet local(0.0);
    for (long k = 0; k < count; ++k) local += field(start + k, a, b);

   +// 4. one datatype + one reduction op, both from the header
   +MPI_Datatype MPI_OTINUM  = oti::mpi::make_datatype<Jet>();
   +MPI_Op       MPI_OTI_SUM = oti::mpi::make_sum_op<Jet>();

   -MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,     MPI_COMM_WORLD);
   +MPI_Allreduce(&local, &global, 1, MPI_OTINUM, MPI_OTI_SUM, MPI_COMM_WORLD);

   +// 5. read the derivatives out of the reduced jet
   +const double d_da = qoi.coeff(oti::sparse({{0, 1}}));   // dQoI/da
   +const double d_db = qoi.coeff(oti::sparse({{1, 1}}));   // dQoI/db
   +//   plus the three second-order coefficients -- the Hessian

What each change is for:

#. **Include the optional headers.** ``otinum/mpi.hpp`` provides *both* the
   datatype and the reduction operator.
#. **Change the scalar type** to a jet carrying the value, gradient, and Hessian.
#. **Seed the design parameters** as variables -- the one new line of intent,
   declaring what you want sensitivities with respect to.
#. **Describe the element and the combine to MPI.** ``make_datatype<Jet>()``
   replaces ``MPI_DOUBLE``; ``make_sum_op<Jet>()`` replaces ``MPI_SUM``. The
   ``Allreduce`` call is otherwise identical.
#. **Read the derivatives out** of the reduced jet with ``coeff()``.

The summation loop and the decomposition are unchanged: the overloaded ``+=``
carries the derivatives through the same code.

The Reduction Operator
----------------------

MPI can ``MPI_SUM`` an ``int`` or a ``double``, but it has no idea how to combine
an ``otinum``. ``otinum/mpi.hpp`` supplies the operator so you do not hand-roll
it:

.. code-block:: cpp

   MPI_Op MPI_OTI_SUM = oti::mpi::make_sum_op<Jet>();   // build + commit
   // ... use in MPI_Reduce / MPI_Allreduce over MPI_OTINUM ...
   oti::mpi::free_op(MPI_OTI_SUM);                       // release

Under the hood ``make_sum_op`` registers (via ``MPI_Op_create``, commutative) a
function of the fixed MPI callback shape:

.. code-block:: cpp

   template <class T>
   void sum_fn(void* in, void* inout, int* len, MPI_Datatype*) {
       const T* a = static_cast<const T*>(in);
       T*       b = static_cast<T*>(inout);
       for (int i = 0; i < *len; ++i) b[i] += a[i];      // otinum::operator+=
   }

That loop is **not** re-defining jet addition: ``b[i] += a[i]`` *is*
``otinum::operator+=``. It is there because MPI's reduction callback is
type-erased and buffer-oriented -- MPI hands the operator two raw ``void*``
buffers of ``*len`` elements at once, so the function casts them to ``Jet*`` and
applies the already-defined ``+=`` across the batch. (``*len`` is 1 for a single
QoI, but MPI may batch many elements per call, so the loop is required.) That
glue is exactly what the header writes once, for every jet shape.

.. note::

   For a **sum**, jet addition is coefficient-wise, so this reduction is
   equivalent to an ``MPI_SUM`` over the ``ncoeffs`` raw doubles of each jet -- you
   could skip the custom op. It is the *general* mechanism: required the moment the
   combine is not coefficient-wise (reducing a **product** of jets, where
   ``operator*`` is a convolution that mixes coefficients), and it keeps the
   reduction expressed in ``MPI_OTINUM`` units.

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
