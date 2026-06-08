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

.. code-block:: cpp

   #include <iostream>
   #include <type_traits>

   #include "otinum/otinum.hpp"

   int main()
   {
       using T = oti::otinum<2, 3, float>;
       static_assert(std::is_same<T::coeff_type, float>::value);

       T x = T::variable(0, 1.25f);
       T y = T::variable(1, 0.5f);
       T f = oti::sin(x) + oti::pow(y + 2.0, 2.0) + 3.0 * x * y;

       std::cout << f.real() << '\n';
       std::cout << f.partial({1, 0}) << '\n';
       std::cout << f.partial({0, 1}) << '\n';
   }

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
