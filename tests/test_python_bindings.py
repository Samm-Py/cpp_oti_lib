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
