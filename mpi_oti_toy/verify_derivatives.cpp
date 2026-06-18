// Verify OTI derivatives against the closed-form analytical derivatives of
// f(x,y) = sin(x) * exp(y), for the four study algebras.
//
// OTI returns EXACT derivatives (not finite-difference approximations), so the
// only discrepancy vs the analytical value is floating-point roundoff -- which is
// exactly what the per-algebra error distribution reveals (double ~1e-15, float
// ~1e-7). This is a property of the algebra and the coefficient type; it does not
// depend on MPI or the number of ranks (MPI transfers the coefficients bit-for-
// bit), so this program is serial.
//
// Emits two CSVs:
//   <out>_errors.csv : algebra,abserr   -- a strided sample of |OTI - analytical|
//                                          over all derivative components (for the
//                                          box plot)
//   <out>_rmse.csv   : algebra,rmse     -- exact RMSE over ALL grid points and
//                                          components (annotation)
//
// Build: g++ -std=c++17 -O2 -I ../include verify_derivatives.cpp -o verify_derivatives
// Run:   ./verify_derivatives /path/to/out_prefix

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "otinum/otinum.hpp"

static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;
static constexpr long SAMPLE_STRIDE = 200;   // ~5000 sampled grid points

// Normalized Taylor coefficient -> partial derivative: d^|a|f/... = coeff * a!.
// (a! = product of factorials of the per-variable orders.)
template <class T>
static double deriv(T const& jet, int dx, int dy)
{
    double fact = 1.0;
    for (int k = 2; k <= dx; ++k) fact *= k;
    for (int k = 2; k <= dy; ++k) fact *= k;
    return static_cast<double>(jet.coeff(oti::sparse(
               dx && dy ? std::initializer_list<oti::sparse_index>{{0, dx}, {1, dy}}
               : dx     ? std::initializer_list<oti::sparse_index>{{0, dx}}
               : dy     ? std::initializer_list<oti::sparse_index>{{1, dy}}
                        : std::initializer_list<oti::sparse_index>{}))) *
           fact;
}

// Analytical derivatives of f = sin(x) e^y.
static double analytic(int dx, int dy, double x, double y)
{
    const double ey = std::exp(y);
    // d/dx cycles sin -> cos -> -sin -> -cos; d/dy leaves e^y unchanged.
    const double sx[4] = {std::sin(x), std::cos(x), -std::sin(x), -std::cos(x)};
    return sx[dx % 4] * ey;
}

template <class T>
static void run_algebra(char const* name, std::FILE* err_f, std::FILE* rmse_f)
{
    const int order = T::order;
    // derivative components present at this order (exclude the value, dx=dy=0)
    std::vector<std::pair<int, int>> comps = {{1, 0}, {0, 1}};
    if (order >= 2) {
        comps.push_back({2, 0});
        comps.push_back({1, 1});
        comps.push_back({0, 2});
    }

    const double h = 1.0 / (N - 1);
    long n_err = 0;
    double sse = 0.0;
    for (long g = 0; g < TOTAL; ++g) {
        const long i = g / N, j = g % N;
        const double x0 = i * h, y0 = j * h;
        T x = T::variable(0, static_cast<typename T::coeff_type>(x0));
        T y = T::variable(1, static_cast<typename T::coeff_type>(y0));
        T f = oti::sin(x) * oti::exp(y);

        const bool sampled = (g % SAMPLE_STRIDE == 0);
        for (auto const& c : comps) {
            const double e =
                std::fabs(deriv(f, c.first, c.second) -
                          analytic(c.first, c.second, x0, y0));
            sse += e * e;
            ++n_err;
            if (sampled) {
                std::fprintf(err_f, "\"%s\",%.17g\n", name, e);
            }
        }
    }
    const double rmse = std::sqrt(sse / static_cast<double>(n_err));
    std::fprintf(rmse_f, "\"%s\",%.17g\n", name, rmse);
    std::printf("  %-16s RMSE = %.3e  (%ld derivative samples)\n", name, rmse,
                n_err);
}

int main(int argc, char** argv)
{
    const std::string prefix = argc > 1 ? argv[1] : "/tmp/mpi_deriv";
    std::FILE* err_f = std::fopen((prefix + "_errors.csv").c_str(), "w");
    std::FILE* rmse_f = std::fopen((prefix + "_rmse.csv").c_str(), "w");
    if (!err_f || !rmse_f) {
        std::fprintf(stderr, "cannot open output files at prefix %s\n",
                     prefix.c_str());
        return 1;
    }
    std::fprintf(err_f, "algebra,abserr\n");
    std::fprintf(rmse_f, "algebra,rmse\n");

    std::printf("OTI derivative accuracy vs analytical (f = sin(x) e^y):\n");
    run_algebra<oti::otinum<2, 1, float>>("<2,1,float>", err_f, rmse_f);
    run_algebra<oti::otinum<2, 1, double>>("<2,1,double>", err_f, rmse_f);
    run_algebra<oti::otinum<2, 2, float>>("<2,2,float>", err_f, rmse_f);
    run_algebra<oti::otinum<2, 2, double>>("<2,2,double>", err_f, rmse_f);

    std::fclose(err_f);
    std::fclose(rmse_f);
    return 0;
}
