#include <array>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "otinum/otinum.hpp"

namespace py = pybind11;

namespace {

template <int M>
std::array<int, M> alpha_from_python(py::sequence seq)
{
    if (py::len(seq) != static_cast<py::ssize_t>(M)) {
        throw py::value_error("multi-index length must match the number of variables");
    }

    std::array<int, M> alpha{};
    for (int i = 0; i < M; ++i) {
        alpha[static_cast<std::size_t>(i)] = seq[static_cast<py::ssize_t>(i)].cast<int>();
    }
    return alpha;
}

template <int M>
bool is_sparse_alpha(py::sequence seq)
{
    if (py::len(seq) == 0) {
        return false;
    }

    for (py::ssize_t i = 0; i < py::len(seq); ++i) {
        py::handle item = seq[i];
        if (!py::isinstance<py::sequence>(item)) {
            return false;
        }
        py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
        if (py::len(pair) != 2) {
            return false;
        }
    }
    return true;
}

template <int M>
std::array<int, M> alpha_from_sparse_python(py::sequence seq)
{
    std::array<int, M> alpha{};
    for (py::ssize_t i = 0; i < py::len(seq); ++i) {
        py::sequence pair = py::reinterpret_borrow<py::sequence>(seq[i]);
        int variable = pair[0].cast<int>();
        int order = pair[1].cast<int>();
        if (variable < 0 || variable >= M) {
            throw py::value_error("sparse multi-index variable out of range");
        }
        if (order < 0) {
            throw py::value_error("sparse multi-index order cannot be negative");
        }
        alpha[static_cast<std::size_t>(variable)] += order;
    }
    return alpha;
}

template <int M>
std::array<int, M> alpha_from_python_any(py::sequence seq)
{
    if (is_sparse_alpha<M>(seq)) {
        return alpha_from_sparse_python<M>(seq);
    }
    return alpha_from_python<M>(seq);
}

template <int M, int N>
std::array<double, oti::otinum<M, N>::ncoeffs> coeffs_from_python(py::sequence seq)
{
    using T = oti::otinum<M, N>;
    if (py::len(seq) != static_cast<py::ssize_t>(T::ncoeffs)) {
        throw py::value_error("coefficient length must match the otinum coefficient count");
    }

    std::array<double, T::ncoeffs> coeffs{};
    for (int i = 0; i < T::ncoeffs; ++i) {
        coeffs[static_cast<std::size_t>(i)] = seq[static_cast<py::ssize_t>(i)].cast<double>();
    }
    return coeffs;
}

template <int M, int N>
std::vector<double> data_as_vector(oti::otinum<M, N> const& value)
{
    auto const& data = value.data();
    return std::vector<double>(data.begin(), data.end());
}

template <int M, int N>
std::string repr(oti::otinum<M, N> const& value, char const* name)
{
    std::ostringstream out;
    out << name << "(real=" << value.real() << ", coeffs=" << value.ncoeffs << ")";
    return out.str();
}

template <int M, int N>
void bind_math_functions(py::module_& m)
{
    using T = oti::otinum<M, N>;

    m.def("exp", [](T const& value) { return oti::exp(value); }, py::arg("value"));
    m.def("log", [](T const& value) { return oti::log(value); }, py::arg("value"));
    m.def("log10", [](T const& value) { return oti::log10(value); }, py::arg("value"));
    m.def("logb", [](T const& value, double base) { return oti::logb(value, base); },
          py::arg("value"), py::arg("base"));
    m.def("pow", [](T const& value, double exponent) { return oti::pow(value, exponent); },
          py::arg("value"), py::arg("exponent"));
    m.def("pow", [](T const& lhs, T const& rhs) { return oti::pow(lhs, rhs); },
          py::arg("lhs"), py::arg("rhs"));
    m.def("sqrt", [](T const& value) { return oti::sqrt(value); }, py::arg("value"));
    m.def("cbrt", [](T const& value) { return oti::cbrt(value); }, py::arg("value"));
    m.def("sin", [](T const& value) { return oti::sin(value); }, py::arg("value"));
    m.def("cos", [](T const& value) { return oti::cos(value); }, py::arg("value"));
    m.def("tan", [](T const& value) { return oti::tan(value); }, py::arg("value"));
    m.def("sinh", [](T const& value) { return oti::sinh(value); }, py::arg("value"));
    m.def("cosh", [](T const& value) { return oti::cosh(value); }, py::arg("value"));
    m.def("tanh", [](T const& value) { return oti::tanh(value); }, py::arg("value"));
    m.def("abs", [](T const& value) { return oti::abs(value); }, py::arg("value"));
    m.def("inv", [](T const& value) { return oti::inv(value); }, py::arg("value"));
    m.def("trunc_mul",
          [](T const& lhs, T const& rhs, int max_order) {
              return oti::trunc_mul(lhs, rhs, max_order);
          },
          py::arg("lhs"), py::arg("rhs"), py::arg("max_order"));
    m.def("trunc_add",
          [](T const& lhs, T const& rhs, int max_order) {
              return oti::trunc_add(lhs, rhs, max_order);
          },
          py::arg("lhs"), py::arg("rhs"), py::arg("max_order"));
    m.def("gem", [](T const& a, T const& b, T const& c) { return oti::gem(a, b, c); },
          py::arg("a"), py::arg("b"), py::arg("c"));
}

template <int M, int N>
void bind_otinum(py::module_& m, char const* name)
{
    using T = oti::otinum<M, N>;

    py::class_<T>(m, name)
        .def(py::init<>())
        .def(py::init<double>(), py::arg("real"))
        .def_static("variable",
                    [](int i, double value) {
                        if (i < 0 || i >= M) {
                            throw py::value_error("variable index out of range");
                        }
                        return T::variable(i, value);
                    },
                    py::arg("i"), py::arg("value") = 0.0)
        .def_static("from_coeffs",
                    [](py::sequence seq) {
                        return T::from_coeffs(coeffs_from_python<M, N>(seq));
                    },
                    py::arg("coeffs"))
        .def_property_readonly_static("nvars", [](py::object) { return M; })
        .def_property_readonly_static("order", [](py::object) { return N; })
        .def_property_readonly_static("ncoeffs", [](py::object) { return T::ncoeffs; })
        .def("real", &T::real)
        .def("data", &data_as_vector<M, N>)
        .def("coeff",
             [](T const& value, py::sequence alpha) {
                 return value.coeff(alpha_from_python_any<M>(alpha));
             },
             py::arg("alpha"))
        .def("set_coeff",
             [](T& value, py::sequence alpha, double coeff) {
                 value.set_coeff(alpha_from_python_any<M>(alpha), coeff);
             },
             py::arg("alpha"), py::arg("coeff"))
        .def("partial",
             [](T const& value, py::sequence alpha) {
                 return value.partial(alpha_from_python_any<M>(alpha));
             },
             py::arg("alpha"))
        .def("set_partial",
             [](T& value, py::sequence alpha, double derivative) {
                 value.set_partial(alpha_from_python_any<M>(alpha), derivative);
             },
             py::arg("alpha"), py::arg("derivative"))
        .def("__len__", [](T const&) { return T::ncoeffs; })
        .def("__getitem__",
             [](T const& value, int i) {
                 if (i < 0) {
                     i += T::ncoeffs;
                 }
                 if (i < 0 || i >= T::ncoeffs) {
                     throw py::index_error("coefficient index out of range");
                 }
                 return value[i];
             })
        .def("__setitem__",
             [](T& value, int i, double coeff) {
                 if (i < 0) {
                     i += T::ncoeffs;
                 }
                 if (i < 0 || i >= T::ncoeffs) {
                     throw py::index_error("coefficient index out of range");
                 }
                 value[i] = coeff;
             })
        .def("__repr__", [name](T const& value) { return repr(value, name); })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + double())
        .def(double() + py::self)
        .def(py::self - double())
        .def(double() - py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self / double())
        .def(double() / py::self)
        .def(-py::self)
        .def("__pow__", [](T const& value, double exponent) { return oti::pow(value, exponent); },
             py::is_operator(), py::arg("exponent"))
        .def("__pow__", [](T const& lhs, T const& rhs) { return oti::pow(lhs, rhs); },
             py::is_operator(), py::arg("rhs"))
        .def("__rpow__",
             [](T const& exponent, double base) {
                 return oti::pow(T(base), exponent);
             },
             py::is_operator(), py::arg("base"));

    bind_math_functions<M, N>(m);
}

} // namespace

PYBIND11_MODULE(otinum, m)
{
    m.doc() = "Python bindings for static OTI numbers";

    bind_otinum<1, 1>(m, "OTI_1_1");
    bind_otinum<1, 2>(m, "OTI_1_2");
    bind_otinum<1, 3>(m, "OTI_1_3");
    bind_otinum<2, 1>(m, "OTI_2_1");
    bind_otinum<2, 2>(m, "OTI_2_2");
    bind_otinum<2, 3>(m, "OTI_2_3");
    bind_otinum<3, 1>(m, "OTI_3_1");
    bind_otinum<3, 2>(m, "OTI_3_2");
    bind_otinum<3, 3>(m, "OTI_3_3");
}
