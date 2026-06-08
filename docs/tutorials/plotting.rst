Plotting Tutorial
=================

The core C++ library does not depend on a plotting package. For documentation
and reports, the clean workflow is to compute values with C++ or the Python
bindings, then render figures with Python and Matplotlib.

Using Existing Python Examples
------------------------------

The repository already contains Python examples that generate figures under
``python_examples/figures``. For example:

.. code-block:: console

   python python_examples/one_dimensional.py
   python python_examples/two_dimensional.py

Those scripts are a good place for visually rich tutorials because they can use
Matplotlib without adding plotting dependencies to the header-only C++ library.

Example Figure
--------------

.. image:: ../../python_examples/figures/one_dimensional/function_and_derivatives.png
   :alt: Function and derivative curves generated from the one-dimensional Python example
   :width: 90%

Minimal Pattern
---------------

The general pattern is:

1. Evaluate OTI values over a grid of input points.
2. Collect real values and derivative values.
3. Plot the arrays with Matplotlib.
4. Save figures into ``python_examples/figures`` or a generated docs output
   directory.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import otinum as oti

   T = oti.OTI_1_3
   xs = np.linspace(-2.0, 2.0, 200)
   values = []
   derivatives = []

   for point in xs:
       x = T.variable(0, float(point))
       y = oti.sin(x) + x * x
       values.append(y.real())
       derivatives.append(y.partial([1]))

   plt.plot(xs, values, label="f(x)")
   plt.plot(xs, derivatives, label="df/dx")
   plt.legend()
   plt.tight_layout()
   plt.savefig("docs/generated/tutorial_plot.png", dpi=160)
