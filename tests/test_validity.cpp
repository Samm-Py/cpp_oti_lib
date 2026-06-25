// Focused unit test for otinum/validity.hpp: surrogate evaluation, truncation
// error, trust check, per-variable reach, and error sensitivity.
//
// Uses a hand-built M=2, P=2 jet with known coefficients so every expected
// number is computed by hand, pinning the model-order / computed-order
// bookkeeping (default model_order = P-1 = 1, the linear surrogate).
//
// Build: c++ -std=c++17 -I ../include test_validity.cpp -o test_validity

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

using J = oti::otinum<2, 2, double>;
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
    std::array<double, 2> const h{0.1, -0.2};

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
    assert(v::is_trusted(jet, std::array<double, 2>{0.2 * 0.99, 0.0}, 0.01));
    assert(!v::is_trusted(jet, std::array<double, 2>{0.2 * 1.01, 0.0}, 0.01));
    // The BOX corner (r0, r1) overshoots the budget -> NOT trusted (box != safe).
    assert(!v::is_trusted(jet, std::array<double, 2>{0.2, 0.1}, 0.01));

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

    std::printf("validity tests passed\n");
    return 0;
}
