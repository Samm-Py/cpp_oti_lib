#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "otinum/otinum.hpp"

namespace {

template <typename F>
double time_loop(int iterations, F&& fn)
{
    auto start = std::chrono::steady_clock::now();
    fn(iterations);
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <int M, int N>
void benchmark_type(int iterations)
{
    using T = oti::otinum<M, N>;

    T x = T::variable(0, 1.2);
    T y = T::variable(M > 1 ? 1 : 0, 0.7);
    T z = T::variable(M > 2 ? 2 : 0, 0.3);
    T sink(0.0);

    auto consume = [&sink](T const& value) {
        sink += value * 1.0e-300;
    };

    double t_add = time_loop(iterations, [&](int n) {
        T local(0.0);
        for (int i = 0; i < n; ++i) {
            local += x + y;
        }
        consume(local);
    });

    double t_mul = time_loop(iterations, [&](int n) {
        T local(0.0);
        for (int i = 0; i < n; ++i) {
            local += x * y;
        }
        consume(local);
    });

    double t_div = time_loop(iterations, [&](int n) {
        T local(0.0);
        T denom = y + 2.0;
        for (int i = 0; i < n; ++i) {
            local += x / denom;
        }
        consume(local);
    });

    double t_exp = time_loop(iterations, [&](int n) {
        T local(0.0);
        for (int i = 0; i < n; ++i) {
            local += oti::exp(x);
        }
        consume(local);
    });

    double t_sin = time_loop(iterations, [&](int n) {
        T local(0.0);
        for (int i = 0; i < n; ++i) {
            local += oti::sin(x + 0.1 * y);
        }
        consume(local);
    });

    double t_mixed = time_loop(iterations, [&](int n) {
        T local(0.0);
        for (int i = 0; i < n; ++i) {
            local += oti::sin(x * y) + oti::exp(x) / (z + 2.0);
        }
        consume(local);
    });

    std::cout << "oti_" << M << "_" << N << ','
              << T::ncoeffs << ','
              << T::table_type::nproducts << ','
              << t_add << ','
              << t_mul << ','
              << t_div << ','
              << t_exp << ','
              << t_sin << ','
              << t_mixed << ','
              << sink.real() << '\n';
}

} // namespace

int main()
{
    int iterations = 200000;

    std::cout << "type,ncoeffs,nproducts,add_s,mul_s,div_s,exp_s,sin_s,mixed_s,sink\n";
    benchmark_type<1, 1>(iterations);
    benchmark_type<1, 2>(iterations);
    benchmark_type<1, 3>(iterations);
    benchmark_type<2, 1>(iterations);
    benchmark_type<2, 2>(iterations);
    benchmark_type<2, 3>(iterations);
    benchmark_type<3, 1>(iterations);
    benchmark_type<3, 2>(iterations);
    benchmark_type<3, 3>(iterations);
}
