#include <cassert>
#include <iostream>
#include <limits>

#include "test_utils.hpp"

namespace {

// axpy and scale_add perform the identical per-coefficient operations as the
// operator chains they replace. They are not compared bit-for-bit because
// floating-point contraction may round them differently: a compiler that
// contracts a*b + c into a hardware fma (Clang does at -O2 by default)
// rounds the fused form once per coefficient but the chain twice. The two
// forms therefore agree to the last ulp, not necessarily exactly.
template <int M, int N, class Coeff>
void check_exact_forms()
{
    using T = oti::otinum<M, N, Coeff>;
    T x = T::variable(0, Coeff(1.5));
    T y = T::variable(M - 1, Coeff(-0.75));
    for (int i = 0; i < T::ncoeffs; ++i) {
        x[i] += Coeff(0.01) * Coeff(i);
        y[i] += Coeff(0.02) * Coeff(i);
    }
    Coeff const s = Coeff(0.375);

    T chained = y + s * x;
    T fused_axpy = y;
    oti::axpy(fused_axpy, s, x);
    T fused_scale_add = oti::scale_add(y, s, x);
    double const ulp_tol = 8.0 * static_cast<double>(std::numeric_limits<Coeff>::epsilon());
    oti_test::expect_all_near(fused_axpy, chained, ulp_tol);
    oti_test::expect_all_near(fused_scale_add, chained, ulp_tol);
}

// fma_into accumulates product terms directly onto y, so it may differ from
// y + a*b in the last ulp; it must agree within tolerance, and must equal
// a*b exactly when y starts at zero.
template <int M, int N, class Coeff>
void check_fma_into()
{
    using T = oti::otinum<M, N, Coeff>;
    T a = T::variable(0, Coeff(2.0));
    T b = T::variable(M - 1, Coeff(3.0));
    T y(Coeff(0.5));
    for (int i = 0; i < T::ncoeffs; ++i) {
        a[i] += Coeff(0.03) * Coeff(i);
        b[i] += Coeff(0.05) * Coeff(i);
        y[i] += Coeff(0.07) * Coeff(i);
    }

    T chained = y + a * b;
    T fused = y;
    oti::fma_into(fused, a, b);
    // Accumulation-order difference is a few ulps of the coefficient type.
    double const ulp_tol = 64.0 * static_cast<double>(std::numeric_limits<Coeff>::epsilon());
    oti_test::expect_all_near(fused, chained, ulp_tol);

    T from_zero{};
    oti::fma_into(from_zero, a, b);
    T product = a * b;
    for (int i = 0; i < T::ncoeffs; ++i) {
        assert(from_zero[i] == product[i]);
    }
}

} // namespace

int main()
{
    check_exact_forms<3, 1, double>();
    check_exact_forms<2, 2, double>();
    check_exact_forms<3, 3, double>();
    check_exact_forms<3, 1, float>();
    check_exact_forms<3, 3, float>();

    check_fma_into<3, 1, double>();
    check_fma_into<2, 2, double>();
    check_fma_into<3, 3, double>();
    check_fma_into<4, 4, double>();
    check_fma_into<3, 1, float>();

    std::cout << "fused operation tests passed\n";
}
