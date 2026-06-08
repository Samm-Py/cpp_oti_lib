#ifndef OTI_ENABLE_KOKKOS
#define OTI_ENABLE_PROFILE
#endif

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "test_utils.hpp"

int main()
{
#ifdef OTI_ENABLE_KOKKOS
    std::cout << "profile counters disabled in Kokkos builds\n";
    return 0;
#else
    using T = oti::otinum<1, 2>;

    oti::profile::reset();
    oti::profile::counters zero = oti::profile::snapshot();
    assert(zero.add == 0);
    assert(zero.mul == 0);
    assert(zero.exp == 0);

    T x = T::variable(0, 1.5);
    T y = T::variable(0, 0.5);

    T add = x + y;
    T add_scalar = add + 1.0;
    T sub = add_scalar - y;
    T sub_scalar = sub - 1.0;
    T neg = -sub_scalar;
    T mul = x * y;
    T mul_scalar = mul * 2.0;
    T div_oti = x / y;
    T div_scalar = x / 2.0;
    T inverse = oti::inv(x);
    T truncated_add = oti::trunc_add(x, y, 1);
    T truncated_mul = oti::trunc_mul(x, y, 1);
    T fused = oti::gem(x, y, T(3.0));
    T elementary = oti::exp(x) + oti::log(x) + oti::pow(x, 2.0) + oti::sin(x) + oti::cos(x) +
                   oti::tan(x) + oti::sinh(x) + oti::cosh(x) + oti::tanh(x) + oti::abs(x);

    double sink = neg.real() + mul_scalar.real() + div_oti.real() + div_scalar.real() +
                  inverse.real() + truncated_add.real() + truncated_mul.real() + fused.real() +
                  elementary.real();
    assert(sink == sink);

    oti::profile::counters c = oti::profile::snapshot();
    assert(c.add > 0 && c.add_oti > 0 && c.add_scalar > 0);
    assert(c.sub > 0 && c.sub_oti > 0 && c.sub_scalar > 0);
    assert(c.neg > 0);
    assert(c.mul > 0 && c.mul_oti > 0 && c.mul_scalar > 0);
    assert(c.div > 0 && c.div_oti > 0 && c.div_scalar > 0);
    assert(c.inv > 0);
    assert(c.trunc_add > 0 && c.trunc_mul > 0 && c.gem > 0);
    assert(c.exp > 0 && c.log > 0 && c.pow > 0);
    assert(c.sin > 0 && c.cos > 0 && c.tan > 0);
    assert(c.sinh > 0 && c.cosh > 0 && c.tanh > 0);
    assert(c.abs > 0);

    std::ostringstream header;
    std::ostringstream row;
    oti::profile::write_csv_header(header);
    oti::profile::write_csv_row(row, "profile_smoke", c);
    assert(header.str().find("run,add,add_oti") == 0);
    assert(row.str().find("profile_smoke,") == 0);

    oti::profile::reset();
    oti::profile::counters reset = oti::profile::snapshot();
    assert(reset.add == 0);
    assert(reset.mul == 0);
    assert(reset.exp == 0);

    std::cout << "profile counter tests passed\n";
    return 0;
#endif
}
