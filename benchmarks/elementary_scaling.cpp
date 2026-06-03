#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "otinum/otinum.hpp"

namespace {

volatile double global_sink = 0.0;

struct ResultRow {
    std::string case_name;
    int order;
    int dimension;
    int ncoeffs;
    int nproducts;
    int trials;
    int double_iterations;
    int oti_iterations;
    double double_seconds_per_eval;
    double double_mean_seconds_per_eval;
    double double_min_seconds_per_eval;
    double double_max_seconds_per_eval;
    double double_stddev_seconds_per_eval;
    double oti_seconds_per_eval;
    double oti_mean_seconds_per_eval;
    double oti_min_seconds_per_eval;
    double oti_max_seconds_per_eval;
    double oti_stddev_seconds_per_eval;
    double ratio;
    double value_real;
};

struct TimingStats {
    double median = 0.0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

template <typename F>
double time_loop(int iterations, F&& fn)
{
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        global_sink += fn().real();
    }
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <typename F>
double time_loop_double(int iterations, F&& fn)
{
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        global_sink += fn();
    }
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

TimingStats summarize(std::vector<double> values)
{
    TimingStats stats;
    std::sort(values.begin(), values.end());

    stats.min = values.front();
    stats.max = values.back();
    if (values.size() % 2 == 0) {
        stats.median = 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
    } else {
        stats.median = values[values.size() / 2];
    }

    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());

    if (values.size() > 1) {
        double sum_sq = 0.0;
        for (double value : values) {
            double diff = value - stats.mean;
            sum_sq += diff * diff;
        }
        stats.stddev = std::sqrt(sum_sq / static_cast<double>(values.size() - 1));
    }

    return stats;
}

double input_value(int i)
{
    return 0.2 + 0.003 * static_cast<double>(i + 1);
}

double weight_a(int i)
{
    return 0.8 + 0.001 * static_cast<double>(i + 1);
}

double weight_b(int i)
{
    return -0.35 + 0.0007 * static_cast<double>(i + 1);
}

double weight_c(int i)
{
    return 0.15 + 0.0003 * static_cast<double>(i + 1);
}

template <int M, int N>
std::vector<oti::otinum<M, N>> make_oti_inputs()
{
    using T = oti::otinum<M, N>;
    std::vector<T> xs;
    xs.reserve(M);
    for (int i = 0; i < M; ++i) {
        xs.push_back(T::variable(i, input_value(i)));
    }
    return xs;
}

template <int M>
std::vector<double> make_double_inputs()
{
    std::vector<double> xs;
    xs.reserve(M);
    for (int i = 0; i < M; ++i) {
        xs.push_back(input_value(i));
    }
    return xs;
}

template <typename T>
T first_order_simple(std::vector<T> const& xs)
{
    T s(0.0);
    T q(0.0);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        s += weight_a(static_cast<int>(i)) * xs[i];
        q += weight_b(static_cast<int>(i)) * xs[i];
    }
    return s + 0.01 * s * q;
}

double first_order_simple(std::vector<double> const& xs)
{
    double s = 0.0;
    double q = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        s += weight_a(static_cast<int>(i)) * xs[i];
        q += weight_b(static_cast<int>(i)) * xs[i];
    }
    return s + 0.01 * s * q;
}

template <int M, int N>
int oti_iterations()
{
    using T = oti::otinum<M, N>;
    long long work = static_cast<long long>(T::ncoeffs) * static_cast<long long>(M + 10) +
                     6LL * static_cast<long long>(T::table_type::nproducts);
    int iterations = static_cast<int>(120000000LL / std::max<long long>(work, 1));
    return std::clamp(iterations, 3, 5000);
}

template <int M>
int double_iterations()
{
    int iterations = static_cast<int>(20000000 / std::max(M, 1));
    return std::clamp(iterations, 20000, 2000000);
}

template <int M, int N, typename OtiFunction, typename DoubleFunction>
ResultRow run_case(std::string const& case_name,
                   OtiFunction&& oti_function,
                   DoubleFunction&& double_function)
{
    using T = oti::otinum<M, N>;
    auto oti_inputs = make_oti_inputs<M, N>();
    auto double_inputs = make_double_inputs<M>();

    int d_iters = double_iterations<M>();
    int o_iters = oti_iterations<M, N>();
    constexpr int trials = 25;

    std::vector<double> double_trials;
    std::vector<double> oti_trials;
    double_trials.reserve(trials);
    oti_trials.reserve(trials);

    for (int trial = 0; trial < trials; ++trial) {
        double d_seconds = time_loop_double(d_iters, [&]() {
            return double_function(double_inputs);
        });
        double o_seconds = time_loop(o_iters, [&]() {
            return oti_function(oti_inputs);
        });

        double_trials.push_back(d_seconds / static_cast<double>(d_iters));
        oti_trials.push_back(o_seconds / static_cast<double>(o_iters));
    }

    TimingStats d_stats = summarize(double_trials);
    TimingStats o_stats = summarize(oti_trials);

    T value = oti_function(oti_inputs);
    return ResultRow{
        case_name,
        N,
        M,
        T::ncoeffs,
        T::table_type::nproducts,
        trials,
        d_iters,
        o_iters,
        d_stats.median,
        d_stats.mean,
        d_stats.min,
        d_stats.max,
        d_stats.stddev,
        o_stats.median,
        o_stats.mean,
        o_stats.min,
        o_stats.max,
        o_stats.stddev,
        o_stats.median / d_stats.median,
        value.real(),
    };
}

template <int M>
void append_dimension_rows(std::vector<ResultRow>& rows)
{
    using T1 = oti::otinum<M, 1>;

    rows.push_back(run_case<M, 1>("first_order_simple",
                                  [](std::vector<T1> const& xs) {
                                      return first_order_simple(xs);
                                  },
                                  [](std::vector<double> const& xs) {
                                      return first_order_simple(xs);
                                  }));
}

void write_csv(std::filesystem::path const& filename, std::vector<ResultRow> const& rows)
{
    std::ofstream ofs(filename);
    ofs << "case,order,dimension,ncoeffs,nproducts,trials,double_iterations,oti_iterations,"
           "double_seconds_per_eval,double_mean_seconds_per_eval,double_min_seconds_per_eval,"
           "double_max_seconds_per_eval,double_stddev_seconds_per_eval,"
           "oti_seconds_per_eval,oti_mean_seconds_per_eval,oti_min_seconds_per_eval,"
           "oti_max_seconds_per_eval,oti_stddev_seconds_per_eval,ratio,value_real\n";
    for (auto const& row : rows) {
        ofs << row.case_name << ','
            << row.order << ','
            << row.dimension << ','
            << row.ncoeffs << ','
            << row.nproducts << ','
            << row.trials << ','
            << row.double_iterations << ','
            << row.oti_iterations << ','
            << row.double_seconds_per_eval << ','
            << row.double_mean_seconds_per_eval << ','
            << row.double_min_seconds_per_eval << ','
            << row.double_max_seconds_per_eval << ','
            << row.double_stddev_seconds_per_eval << ','
            << row.oti_seconds_per_eval << ','
            << row.oti_mean_seconds_per_eval << ','
            << row.oti_min_seconds_per_eval << ','
            << row.oti_max_seconds_per_eval << ','
            << row.oti_stddev_seconds_per_eval << ','
            << row.ratio << ','
            << row.value_real << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path output = "benchmarks/results/elementary_scaling.csv";
    if (argc > 1) {
        output = argv[1];
    }

    std::filesystem::create_directories(output.parent_path());

    std::vector<ResultRow> rows;
    append_dimension_rows<1>(rows);
    append_dimension_rows<2>(rows);
    append_dimension_rows<5>(rows);
    append_dimension_rows<10>(rows);
    append_dimension_rows<20>(rows);
    append_dimension_rows<30>(rows);

    write_csv(output, rows);

    std::cout << "wrote " << output << '\n';
    std::cout << "global sink: " << global_sink << '\n';
    for (auto const& row : rows) {
        std::cout << row.case_name
                  << " M=" << row.dimension
                  << " N=" << row.order
                  << " ratio=" << row.ratio
                  << " oti_iters=" << row.oti_iterations
                  << '\n';
    }
}
