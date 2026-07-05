// Trust-region basics for a jet used as a surrogate model.
//
// f(x, y) = exp(x) + exp(y) + x*y/2 is expanded at (0, 0) into an
// otinum<2, 2> jet: one order ABOVE the linear model we intend to trust, so
// the order-2 coefficients are available as the truncation-error estimate.
// The oti::validity primitives then answer, without re-evaluating f:
//   - what does the linear surrogate predict at a step h?
//   - how large is the estimated error of that prediction?
//   - is the prediction within a relative tolerance tau?
//   - how far can each variable move on its own before it is not?
//   - which variable do I pull back when it is not?
//
// See docs/tutorials/validity.rst for the walk-through.
#include <cstdio>

#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

int main()
{
    using T = oti::otinum<2, 2>;
    namespace v = oti::validity;

    T x = T::variable(0, 0.0);
    T y = T::variable(1, 0.0);
    T f = oti::exp(x) + oti::exp(y) + 0.5 * x * y;
    // Jet at (0, 0): value 2, gradient (1, 1), stored second-order
    // coefficients c20 = 0.5, c11 = 0.5, c02 = 0.5.

    double const tau = 0.05;                     // 5% relative tolerance
    oti::detail::array<double, 2> h{0.3, -0.2};  // candidate step

    std::printf("value                 = %g\n", f.real());
    std::printf("linear prediction     = %g\n", v::evaluate(f, h));
    std::printf("estimated error       = %g\n", v::truncation_error(f, h));
    std::printf("budget (tau*|f|)      = %g\n", tau * f.real());
    std::printf("trusted at h?         = %s\n",
                v::is_trusted(f, h, tau) ? "yes" : "no");

    auto r = v::validity_radius(f, tau);
    std::printf("per-axis reach        = (%g, %g)\n", r[0], r[1]);

    // The reaches are single-axis statements. Stepping to the BOX CORNER
    // (r0, r1) moves both variables to their individual limits at once and
    // overshoots the budget -- always re-check a combined step.
    oti::detail::array<double, 2> corner{r[0], r[1]};
    std::printf("trusted at corner?    = %s\n",
                v::is_trusted(f, corner, tau) ? "yes" : "no");
    std::printf("error at corner       = %g\n",
                v::truncation_error(f, corner));

    auto g = v::error_sensitivity(f, h);
    std::printf("error sensitivity     = (%g, %g)\n", g[0], g[1]);
    return 0;
}
