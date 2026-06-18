Derivative Accuracy
===================

Before distributing anything, it is worth establishing what the answer *is* and
that moving it around does not change it. OTI returns **exact** derivatives -- it
propagates the chain rule analytically, not by finite differences -- so the only
discrepancy from the closed-form derivative is floating-point roundoff. This is a
property of the OTI algebra and the coefficient type; it has nothing to do with
how many ranks or which devices you use.

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

.. note::

   Because this accuracy is a property of the **algebra**, not the
   parallelization, and MPI transfers coefficients bit-for-bit (the round-trip
   test in :doc:`make_datatype` confirms this), the box plot is identical at one
   rank or a thousand, on CPU or GPU. **Distributing the work changes how fast you
   get the answer, not what the answer is** -- which is why the rest of this
   section is about speed and execution, never correctness.
