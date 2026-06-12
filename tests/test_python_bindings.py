import math

import otinum as oti


def assert_close(actual, expected, tolerance=1e-11):
    assert abs(actual - expected) <= tolerance


def assert_all_close(actual, expected, tolerance=1e-11):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert_close(actual[i], expected[i], tolerance)


def test_bound_type_metadata_and_accessors():
    T = oti.OTI_2_3
    assert T.nvars == 2
    assert T.order == 3
    assert T.ncoeffs == 10

    x = T.variable(0, 1.5)
    assert_close(x.real(), 1.5)
    assert_close(x.partial([1, 0]), 1.0)
    assert_close(x.partial([0, 1]), 0.0)
    assert len(x.data()) == T.ncoeffs

    value = T.from_coeffs([0.1 * i for i in range(T.ncoeffs)])
    assert_close(value[3], 0.3)
    value[-1] = 2.5
    assert_close(value[T.ncoeffs - 1], 2.5)

    value.set_coeff([2, 1], 4.0)
    assert_close(value.coeff([2, 1]), 4.0)
    assert_close(value.partial([2, 1]), 8.0)


def test_sparse_multi_indices_in_python():
    T = oti.OTI_3_3
    x0 = T.variable(0, 1.5)
    x2 = T.variable(2, 2.0)
    product = x0 * x2

    assert_close(product.partial([[0, 1]]), 2.0)
    assert_close(product.partial([[2, 1]]), 1.5)
    assert_close(product.partial([[0, 1], [2, 1]]), 1.0)
    assert_close(product.partial([[0, 1], [0, 1], [2, 1]]), 0.0)


def test_python_math_matches_cpp_examples():
    T = oti.OTI_2_3
    x0 = 1.5
    y0 = 0.3
    x = T.variable(0, x0)
    y = T.variable(1, y0)

    f = oti.sin(x * y) + oti.exp(x)
    xy = x0 * y0

    assert_close(f.real(), math.sin(xy) + math.exp(x0))
    assert_close(f.partial([1, 0]), y0 * math.cos(xy) + math.exp(x0))
    assert_close(f.partial([0, 1]), x0 * math.cos(xy))
    assert_close(f.partial([1, 1]), math.cos(xy) - xy * math.sin(xy))

    assert_all_close(oti.pow(x, 2.0), x * x)
    assert_all_close(oti.sqrt(x * x), x)
    assert_all_close(oti.log(oti.exp(x)), x, 1e-10)
    assert_all_close(oti.tanh(x), oti.sinh(x) / oti.cosh(x), 1e-12)


def test_comparisons_use_real_part_only():
    T = oti.OTI_2_2
    x = T.variable(0, 0.5)
    y = T.variable(1, 1.2)

    assert x < y and y > x and x <= y and y >= x and x != y
    assert T(2.0) == T(2.0)
    # Equality ignores derivative coefficients, matching the C++ operators.
    assert T.variable(0, 2.0) == T(2.0)

    assert x == 0.5 and 0.5 == x
    assert x < 1.0 and 1.0 > x and x >= 0.5 and 0.4 <= x and x != 0.6


def test_inverse_trig_functions():
    T = oti.OTI_2_2
    x = T.variable(0, 0.5)
    y = T.variable(1, 1.2)

    f = oti.asin(x)
    assert_close(f.real(), math.asin(0.5))
    assert_close(f.partial([1, 0]), 1.0 / math.sqrt(1.0 - 0.25))
    assert_close(oti.acos(x).partial([1, 0]), -1.0 / math.sqrt(1.0 - 0.25))
    assert_close(oti.atan(x).partial([1, 0]), 1.0 / 1.25)

    g = oti.atan2(y, x)
    assert_close(g.real(), math.atan2(1.2, 0.5))
    assert_close(oti.atan2(y, 0.5).real(), math.atan2(1.2, 0.5))
    assert_close(oti.atan2(1.2, x).real(), math.atan2(1.2, 0.5))
    # d/dy atan2(y, x) = x / (x^2 + y^2)
    assert_close(g.partial([0, 1]), 0.5 / (0.25 + 1.44))


def test_interop_math_surface():
    T = oti.OTI_2_2
    x = T.variable(0, 0.5)

    assert oti.floor(T(2.7)).real() == 2.0
    assert oti.ceil(T(2.1)).real() == 3.0
    assert oti.trunc(T(-2.7)).real() == -2.0
    assert oti.round(T(2.5)).real() == 3.0
    assert oti.fabs(T(-2.0)).real() == 2.0

    assert oti.isnan(T(float("nan"))) and not oti.isnan(x)
    assert oti.isinf(T(float("inf"))) and not oti.isinf(x)
    assert oti.isfinite(x) and not oti.isfinite(T(float("inf")))
    assert oti.signbit(T(-1.0)) and not oti.signbit(x)

    assert oti.copysign(x, -2.0).real() == -0.5
    assert oti.fmax(x, 0.9).real() == 0.9
    assert oti.fmin(0.1, x).real() == 0.1
    assert_close(oti.hypot(x, 1.2).real(), math.hypot(0.5, 1.2))
    assert_close(oti.fmod(T(5.5), 2.0).real(), 1.5)
    assert_close(oti.log1p(x).real(), math.log1p(0.5))
    assert_close(oti.expm1(x).real(), math.expm1(0.5))
    assert_close(oti.log2(x).real(), math.log2(0.5))
    assert_close(oti.exp2(x).real(), 2.0**0.5)


def test_fused_ops_match_operator_chains():
    T = oti.OTI_2_2
    x = T.variable(0, 0.5)
    y = T.variable(1, 1.2)

    a = oti.exp(x)
    reference = a + 0.25 * y
    assert_all_close(oti.scale_add(a, 0.25, y), reference, 0.0)

    in_place = oti.exp(x)
    oti.axpy(in_place, 0.25, y)
    assert_all_close(in_place, reference, 0.0)

    accumulator = T(1.0)
    oti.fma_into(accumulator, x, y)
    assert_all_close(accumulator, T(1.0) + x * y, 1e-15)


def test_python_domain_edges_propagate_nan_derivatives():
    T = oti.OTI_1_2

    log_bad = oti.log(T.variable(0, -1.0))
    assert math.isnan(log_bad.real())
    assert math.isnan(log_bad.partial([1]))
    assert math.isnan(log_bad.partial([2]))

    inv_zero = oti.inv(T.variable(0, 0.0))
    assert math.isinf(inv_zero.real())
    assert math.isnan(inv_zero.partial([1]))
    assert math.isnan(inv_zero.partial([2]))

    sqrt_zero = oti.sqrt(T.variable(0, 0.0))
    assert_close(sqrt_zero.real(), 0.0)
    assert math.isnan(sqrt_zero.partial([1]))
    assert math.isnan(sqrt_zero.partial([2]))
