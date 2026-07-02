// Focused unit test for otinum/validity.hpp: surrogate evaluation, truncation
// error, trust check, per-variable reach, and error sensitivity.
//
// Uses a hand-built M=2, P=2 jet with known coefficients so every expected
// number is computed by hand, pinning the model-order / computed-order
// bookkeeping (default model_order = P-1 = 1, the linear surrogate).
//
// Build: c++ -std=c++17 -I ../include test_validity.cpp -o test_validity

// Keep the asserts alive in NDEBUG (Release ctest) builds; see the note in
// test_utils.hpp for why this is done in-source rather than with -UNDEBUG.
#undef NDEBUG

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

using J = oti::otinum<2, 2, double>;
// Step type matching the library's array (std::array on host, Kokkos::Array in
// OTI_ENABLE_KOKKOS builds -- std::array does not convert to Kokkos::Array, so
// spelling std::array here would not compile in a Kokkos build).
using arr1 = oti::detail::array<double, 1>;
using arr2 = oti::detail::array<double, 2>;
using A = J::alpha_type;
namespace v = oti::validity;

static bool close(double a, double b, double tol = 1e-12)
{
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

// f(x,y) surrogate: value 2, gradient (3,-1), and normalized 2nd-order coeffs
//   c_20 = 0.5, c_11 = 0.25, c_02 = 2.0   (c_ab is the stored Taylor coeff).
static J make_jet()
{
    J jet{};
    jet.set_coeff(A{{0, 0}}, 2.0);
    jet.set_coeff(A{{1, 0}}, 3.0);
    jet.set_coeff(A{{0, 1}}, -1.0);
    jet.set_coeff(A{{2, 0}}, 0.5);
    jet.set_coeff(A{{1, 1}}, 0.25);
    jet.set_coeff(A{{0, 2}}, 2.0);
    return jet;
}

int main()
{
    J const jet = make_jet();
    arr2 const h{0.1, -0.2};

    // --- evaluate: linear (default m=1) and full quadratic (m=2) ---
    // linear:    2 + 3*0.1 + (-1)*(-0.2) = 2.5
    assert(close(v::evaluate(jet, h), 2.5));
    // quadratic: 2.5 + (0.5*0.01 + 0.25*(-0.02) + 2.0*0.04) = 2.5 + 0.08 = 2.58
    assert(close(v::evaluate(jet, h, 2), 2.58));
    // m=0 is just the value.
    assert(close(v::evaluate(jet, h, 0), 2.0));

    // --- truncation error of the linear model: order-2 part = 0.08 ---
    assert(close(v::truncation_error(jet, h), 0.08));

    // --- is_trusted: budget = tau*|f| ---
    assert(!v::is_trusted(jet, h, 0.01));  // 0.08 > 0.02
    assert(v::is_trusted(jet, h, 0.05));   // 0.08 <= 0.10

    // --- validity_radius (tau=0.01, budget=0.02): r0=sqrt(0.02/0.5)=0.2,
    //     r1=sqrt(0.02/2.0)=0.1 ---
    auto r = v::validity_radius(jet, 0.01);
    assert(close(r[0], 0.2));
    assert(close(r[1], 0.1));

    // Bracket the reach r0=0.2 along axis 0: just inside is trusted, just outside
    // is not (exact-boundary equality is FP knife-edge, so we straddle it).
    assert(v::is_trusted(jet, arr2{0.2 * 0.99, 0.0}, 0.01));
    assert(!v::is_trusted(jet, arr2{0.2 * 1.01, 0.0}, 0.01));
    // The BOX corner (r0, r1) overshoots the budget -> NOT trusted (box != safe).
    assert(!v::is_trusted(jet, arr2{0.2, 0.1}, 0.01));

    // --- error_sensitivity (gradient of E = 0.5 x^2 + 0.25 xy + 2 y^2) ---
    // g0 = 2*0.5*0.1 + 0.25*(-0.2) = 0.05 ;  g1 = 0.25*0.1 + 2*2.0*(-0.2) = -0.775
    auto g = v::error_sensitivity(jet, h);
    assert(close(g[0], 0.05));
    assert(close(g[1], -0.775));

    // --- zero pure term => infinite reach along that axis ---
    J flat = jet;
    flat.set_coeff(A{{2, 0}}, 0.0);  // no pure x^2 curvature
    auto rf = v::validity_radius(flat, 0.01);
    assert(std::isinf(rf[0]));
    assert(close(rf[1], 0.1));

    // ===== multi-band fold: a <1,3> jet certifying the LINEAR model (m=1), so
    // the order-2 AND order-3 bands both feed the error =====
    {
        using J3 = oti::otinum<1, 3, double>;
        using A3 = J3::alpha_type;
        J3 g3{};
        g3.set_coeff(A3{{0}}, 2.0);  // c0
        g3.set_coeff(A3{{1}}, 0.0);  // c1
        g3.set_coeff(A3{{2}}, 1.0);  // c2
        g3.set_coeff(A3{{3}}, 0.5);  // c3
        arr1 const h3{0.1};

        // truncation_error (m=1) folds orders 2 AND 3: c2 h^2 + c3 h^3
        //   = 1*0.01 + 0.5*0.001 = 0.0105
        assert(close(v::truncation_error(g3, h3, 1), 0.0105));
        // identical to full(m=3) - linear(m=1) prediction
        assert(close(v::truncation_error(g3, h3, 1),
                     v::evaluate(g3, h3, 3) - v::evaluate(g3, h3, 1)));
        // the single leading band alone would be only c2 h^2 = 0.01 -> the fold
        // genuinely adds the order-3 contribution
        assert(v::truncation_error(g3, h3, 1) > 0.01 + 1e-9);

        // error_sensitivity (m=1) folds both bands: 2 c2 h + 3 c3 h^2
        //   = 0.2 + 3*0.5*0.01 = 0.215
        assert(close(v::error_sensitivity(g3, h3, 1)[0], 0.215));

        // validity_radius (m=1): root of |c2 r^2 + c3 r^3| = budget (no closed
        // form with two bands -> bracket-and-bisection).
        double const tau3 = 0.005, budget3 = tau3 * 2.0;  // 0.01
        auto r3 = v::validity_radius(g3, tau3, 0.0, 1);
        // residual: the pure-axis error at the returned r equals the budget
        double const rr = r3[0];
        assert(close(rr * rr + 0.5 * rr * rr * rr, budget3, 1e-9));
        // tighter than the leading-order-only closed form sqrt(budget/c2) = 0.1
        assert(rr < 0.1);
        // consistent with is_trusted straddling the boundary along the axis
        assert(v::is_trusted(g3, arr1{rr * 0.99}, tau3, 0.0, 1));
        assert(!v::is_trusted(g3, arr1{rr * 1.01}, tau3, 0.0, 1));
    }

    // ===== absolute floor tau_abs: a signed QoI crossing zero =====
    // f(x) = x + x^2 at x0 = 0: real part 0, so the relative budget tau*|f|
    // collapses and nothing is trusted; tau_abs restores a meaningful check.
    {
        using J0 = oti::otinum<1, 2, double>;
        using A0 = J0::alpha_type;
        J0 z{};
        z.set_coeff(A0{{0}}, 0.0);  // f = 0 exactly
        z.set_coeff(A0{{1}}, 1.0);
        z.set_coeff(A0{{2}}, 1.0);  // truncation error of the linear model: h^2
        arr1 const hz{0.05};  // error 0.0025

        // Relative-only: budget = 0 -> untrusted at any step, radius 0.
        assert(!v::is_trusted(z, hz, 0.01));
        assert(close(v::validity_radius(z, 0.01)[0], 0.0));

        // With an absolute floor: budget = tau_abs = 0.01 -> trusted for
        // |h| <= sqrt(0.01) = 0.1, and the radius is that closed form.
        assert(v::is_trusted(z, hz, 0.01, 0.01));
        assert(!v::is_trusted(z, arr1{0.2}, 0.01, 0.01));
        assert(close(v::validity_radius(z, 0.01, 0.01)[0], 0.1));

        // Away from zero both terms contribute: budget = tau_abs + tau*|f|.
        J0 nz = z;
        nz.set_coeff(A0{{0}}, 2.0);
        assert(close(v::validity_radius(nz, 0.01, 0.01)[0],
                     std::sqrt(0.01 + 0.01 * 2.0)));
    }

    std::printf("validity tests passed\n");
    return 0;
}
