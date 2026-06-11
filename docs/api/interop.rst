Standard-Library Interoperability
=================================

``otinum/otinum.hpp`` includes the interoperability layer from
``otinum/interop.hpp``. It provides standard-library-oriented operations that
help generic scalar code continue to compile when its scalar type changes from
``double`` to ``oti::otinum``.

Use the ``oti`` namespace explicitly in ordinary code:

.. code-block:: cpp

   #include <limits>

   #include "otinum/otinum.hpp"

   using T = oti::otinum<1, 2>;

   T x = T::variable(0, 1.25);
   T rounded = oti::floor(x);
   bool valid = oti::isfinite(x);
   T epsilon = std::numeric_limits<T>::epsilon();

In code that is generic over the scalar type, use the standard
using-declaration idiom instead:

.. code-block:: cpp

   template <class Scalar>
   Scalar wrap(Scalar x, Scalar period)
   {
       using std::floor;
       return x - period * floor(x / period);
   }

When ``Scalar`` is an OTI type, the unqualified call finds the ``oti``
overload through argument-dependent lookup; when it is ``double`` or
``float``, the using-declaration provides the standard function. This is the
pattern that lets existing templated numerical code switch its scalar type
from ``double`` to ``oti::otinum`` without edits.

Numeric Limits
--------------

``std::numeric_limits<oti::otinum<M, N, Coeff>>`` forwards scalar properties
from ``std::numeric_limits<Coeff>``. Value-returning members such as ``min()``,
``max()``, ``lowest()``, ``epsilon()``, ``infinity()``, and ``quiet_NaN()``
return constant OTI values whose derivative coefficients are zero.

``is_iec559`` is intentionally false. An OTI value is not a bit-compatible
IEEE-754 scalar even when its coefficient type is ``double``.

Streaming And Inspection
------------------------

Stream insertion prints only the real coefficient. This keeps existing logs and
CSV output scalar-shaped:

.. code-block:: cpp

   std::cout << x; // prints x.real()

Use ``oti::print_coeffs`` when derivative coefficients should also be visible:

.. code-block:: cpp

   oti::print_coeffs(std::cout, x);

The derivative entries are printed in flat normalized-coefficient order.

Rounding And Step Functions
---------------------------

The functions ``floor``, ``ceil``, ``trunc``, ``round``, ``nearbyint``, and
``rint`` apply the corresponding scalar operation to ``real()`` and return zero
derivative coefficients. These functions are piecewise constant away from their
jump points; at a jump the mathematical derivative is undefined, but the
implementation still returns zero derivatives.

``fabs`` forwards to ``oti::abs`` and therefore uses the documented ``abs``
behavior at zero.

Classification Predicates
-------------------------

``signbit``, ``isnan``, ``isinf``, and ``isfinite`` inspect only the real
coefficient and return ``bool``. They are intended for ordinary guard code:

.. code-block:: cpp

   if (!oti::isfinite(x)) {
       // Handle an invalid scalar value.
   }

Selection And Sign Handling
---------------------------

``fmin`` and ``fmax`` select operands by real coefficient and return the selected
operand as a whole, preserving its derivative coefficients. Their NaN behavior
matches the scalar functions, and exact ties select the first argument.

``copysign(x, y)`` applies the sign of ``y.real()`` to ``abs(x)``. Derivatives
therefore flow through ``x``; ``y`` contributes only the branch decision.

These functions are nondifferentiable where the selected branch changes. As
with comparisons, the result contains the derivatives of the branch chosen at
the current real values.

Norms And Remainders
--------------------

``hypot`` supports two or three OTI arguments and mixed two-argument
OTI/scalar calls. It is evaluated as ``sqrt(x*x + y*y [+ z*z])``. This preserves
derivatives but, unlike a carefully scaled scalar ``std::hypot`` implementation,
can overflow for extreme input magnitudes.

``fmod`` and ``remainder`` freeze their integer quotient from the operands'
real coefficients, then evaluate ``x - n*y``. Between quotient jumps this gives
the expected derivatives:

.. code-block:: cpp

   T x = T::variable(0, 7.3);
   T r = oti::fmod(x, 2.0);

   // Away from a quotient jump:
   // r.partial({1}) == 1

The derivative is not defined at points where the integer quotient changes.

Additional Exponential And Logarithmic Functions
------------------------------------------------

The interoperability layer provides:

``exp2``
   Computes ``2**x`` through the analytic OTI exponential path.

``log2``
   Computes the base-two logarithm through the analytic OTI logarithm path.

``expm1``
   Propagates derivatives as ``exp(x) - 1`` while using the dedicated scalar
   function for an accurate real coefficient near zero.

``log1p``
   Propagates derivatives as ``log(1 + x)`` while using the dedicated scalar
   function for an accurate real coefficient near zero.

Generated Reference
-------------------

.. doxygenfile:: interop.hpp
   :sections: innerclass func
