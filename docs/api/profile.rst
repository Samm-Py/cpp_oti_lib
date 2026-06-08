Profiling Helpers
=================

Optional host-side operation profiling is defined in
``include/otinum/profile.hpp``.

Enabling Profiling
------------------

Define ``OTI_ENABLE_PROFILE`` when compiling host-only code. Profiling is
disabled in Kokkos mode because device-callable functions cannot update
host-side global counters.

.. code-block:: console

   c++ -std=c++17 \
     -DOTI_ENABLE_PROFILE \
     -I /root/Research/cpp_oti_lib/include \
     profile_example.cpp \
     -o profile_example

Example
-------

.. code-block:: cpp

   #include <iostream>

   #include "otinum/otinum.hpp"
   #include "otinum/profile.hpp"

   int main()
   {
       using T = oti::otinum<2, 2>;

       oti::profile::reset();

       T x = T::variable(0, 1.5);
       T y = T::variable(1, 0.3);
       T f = oti::sin(x * y) + oti::exp(x);

       auto counts = oti::profile::snapshot();
       oti::profile::write_csv_header(std::cout);
       oti::profile::write_csv_row(std::cout, "example", counts);
   }

The counters are intentionally coarse. They are useful for comparing algorithm
paths and checking whether an expression is dominated by multiplication,
division/inversion, or elementary function composition.

Generated Reference
-------------------

.. doxygenfile:: profile.hpp
   :sections: innerclass func define
