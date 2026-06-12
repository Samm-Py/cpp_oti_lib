Basic C++ Usage
===============

This tutorial evaluates a scalar expression with two OTI variables and extracts
first- and second-order derivatives.

Program
-------

The same source is available in the repository as
``examples/basic_usage.cpp``.

.. literalinclude:: ../../examples/basic_usage.cpp
   :language: cpp

Compile And Run
---------------

From the repository root:

.. code-block:: console

   c++ -std=c++17 -I include examples/basic_usage.cpp -o /tmp/basic_usage
   /tmp/basic_usage

Output
------

The analytic values and OTI values should agree to roundoff:

.. code-block:: text

          f analytic=         4.91665 ad=         4.91665 abs_diff=0
      df/dx analytic=         4.75182 ad=         4.75182 abs_diff=0
      df/dy analytic=         1.35067 ad=         1.35067 abs_diff=0
    d2f/dx2 analytic=         4.44254 ad=         4.44254 abs_diff=0
   d2f/dxdy analytic=        0.704713 ad=        0.704713 abs_diff=1.11022e-16
    d2f/dy2 analytic=       -0.978672 ad=       -0.978672 abs_diff=2.22045e-16

How It Works
------------

``T::variable(0, 1.5)`` creates a value whose real part is ``1.5`` and whose
first derivative with respect to variable 0 is one. All arithmetic and
elementary functions propagate the truncated Taylor coefficients. Calling
``partial({1, 0})`` returns the ordinary derivative with respect to the first
variable.

Use ``coeff(alpha)`` when you need the normalized stored coefficient instead of
the ordinary derivative.
