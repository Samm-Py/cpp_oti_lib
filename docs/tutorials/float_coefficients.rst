Float Coefficients
==================

The default OTI coefficient type is ``double``:

.. code-block:: cpp

   using T = oti::otinum<2, 3>;

For single-precision storage and scalar math, provide ``float`` as the third
template argument:

.. code-block:: cpp

   using T = oti::otinum<2, 3, float>;

Example
-------

The same source is available in the repository as
``examples/float_coefficients.cpp``.

.. literalinclude:: ../../examples/float_coefficients.cpp
   :language: cpp

Compile And Run
---------------

From the repository root:

.. code-block:: console

   c++ -std=c++17 -I include examples/float_coefficients.cpp -o /tmp/float_coefficients
   /tmp/float_coefficients

Output
------

The program prints the real value followed by the first partial derivatives
with respect to ``x`` and ``y``:

.. code-block:: text

   9.07398
   1.81532
   8.75

The ``f`` suffixes on the literals are stylistic, not required: scalar
operands are converted to the coefficient type, so ``3.0 * x * y`` compiles
and produces the same single-precision result as ``3.0f * x * y``.

Why Use Float?
--------------

``float`` can reduce memory traffic and improve throughput on hardware where
single-precision arithmetic is faster than double precision. This is most
useful for larger ``M`` and ``N`` combinations, where each OTI value stores many
coefficients.

Tradeoffs
---------

``float`` has less precision and a smaller dynamic range than ``double``. Use
it when the target problem tolerates single-precision derivative values, and
prefer ``double`` when numerical error is the main concern.
