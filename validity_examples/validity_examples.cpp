// Small analytic intuition-builders for otinum/validity.hpp.
//
// Each example builds an OTI jet of a KNOWN function by evaluating it in OTI
// arithmetic at an expansion point, then exercises evaluate / truncation_error /
// is_trusted / validity_radius / error_sensitivity and writes CSVs the companion
// plot_validity_examples.py turns into figures. Everything is hand-checkable.
//
// Build: c++ -std=c++17 -I ../include validity_examples.cpp -o validity_examples
// Run:   ./validity_examples        # writes data/*.csv
//        python plot_validity_examples.py

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <sys/stat.h>

#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

namespace val = oti::validity;

static std::FILE* open_csv(std::string const& name)
{
    return std::fopen(("data/" + name).c_str(), "w");
}

// Append a one-line record of (tau, f0, reach) so the plotter can draw the
// trusted ellipse/box without recomputing.
static void write_params(char const* name, double tau, double f0, double rx, double ry)
{
    std::FILE* fp = std::fopen("data/params.csv", "a");
    std::fprintf(fp, "%s,%.6g,%.6g,%.6g,%.6g\n", name, tau, f0, rx, ry);
    std::fclose(fp);
}

// ---- Example 1: 1D exp(x) around 0 -- the hello world ----------------------
static void example1()
{
    using J = oti::otinum<1, 2, double>;
    J const f = oti::exp(J::variable(0, 0.0));  // coeffs [1, 1, 1/2]
    double const tau = 0.05;
    double const r = val::validity_radius(f, tau)[0];
    std::printf("[ex1] exp(x) @0: coeffs c0=%.3f c1=%.3f c2=%.3f  reach(tau=%.2f)=%.4f"
                "  (hand: sqrt(2*tau)=%.4f)\n",
                f[0], f[1], f[2], tau, r, std::sqrt(2 * tau));

    std::FILE* fp = open_csv("ex1.csv");
    std::fprintf(fp, "h,truth,linear,real_err,est_err,trusted,reach\n");
    for (int k = -100; k <= 100; ++k) {
        double h = 0.01 * k;
        std::array<double, 1> step{h};
        double lin = val::evaluate(f, step);
        double truth = std::exp(h);
        std::fprintf(fp, "%.4f,%.6g,%.6g,%.6g,%.6g,%d,%.6f\n", h, truth, lin, truth - lin,
                     val::truncation_error(f, step), val::is_trusted(f, step, tau) ? 1 : 0, r);
    }
    std::fclose(fp);
}

// ---- Example 2: reach responds to curvature / nonlinearity -----------------
static void example2()
{
    double const tau = 0.05;

    // (a) exp(k x) @0: reach should scale like sqrt(2 tau)/k.
    std::FILE* fa = open_csv("ex2_curvature.csv");
    std::fprintf(fa, "k,reach,reach_times_k\n");
    std::printf("[ex2a] exp(k x): reach ~ sqrt(2 tau)/k\n");
    for (double k : {0.5, 1.0, 2.0, 4.0, 8.0}) {
        using J = oti::otinum<1, 2, double>;
        J const f = oti::exp(J::variable(0, 0.0) * k);
        double r = val::validity_radius(f, tau)[0];
        std::fprintf(fa, "%.3f,%.6f,%.6f\n", k, r, r * k);
        std::printf("        k=%.1f  reach=%.4f  reach*k=%.4f\n", k, r, r * k);
    }
    std::fclose(fa);

    // (b) 1/(1-x) expanded near its singularity at x=1: reach collapses as x0->1.
    std::FILE* fb = open_csv("ex2_singularity.csv");
    std::fprintf(fb, "x0,distance_to_sing,reach\n");
    std::printf("[ex2b] 1/(1-x): reach collapses approaching the pole at x=1\n");
    for (double x0 : {0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95}) {
        using J = oti::otinum<1, 2, double>;
        J const f = 1.0 / (1.0 - J::variable(0, x0));
        double r = val::validity_radius(f, tau)[0];
        std::fprintf(fb, "%.3f,%.3f,%.6f\n", x0, 1.0 - x0, r);
        std::printf("        x0=%.2f  (dist %.2f)  reach=%.4f\n", x0, 1.0 - x0, r);
    }
    std::fclose(fb);
}

// Write a 2D (hx,hy) grid of real vs estimated error and the trust flag.
template <class J, class TrueFn>
static void write_grid(std::string const& name, J const& f, double tau, double span, TrueFn truth)
{
    std::FILE* fp = open_csv(name);
    std::fprintf(fp, "hx,hy,real_err,est_err,trusted\n");
    for (int iy = -60; iy <= 60; ++iy) {
        for (int ix = -60; ix <= 60; ++ix) {
            double hx = span * ix / 60.0, hy = span * iy / 60.0;
            std::array<double, 2> step{hx, hy};
            double lin = val::evaluate(f, step);
            std::fprintf(fp, "%.5f,%.5f,%.6g,%.6g,%d\n", hx, hy, truth(hx, hy) - lin,
                         val::truncation_error(f, step), val::is_trusted(f, step, tau) ? 1 : 0);
        }
    }
    std::fclose(fp);
}

// ---- Example 3: coupling -- the tilted ellipse and box-corner overshoot ----
static void example3()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = oti::exp(x) + oti::exp(y) + 0.5 * (x * y);  // coupled, definite Hessian
    double const tau = 0.05;
    auto r = val::validity_radius(f, tau);
    std::array<double, 2> corner{r[0], r[1]}, axis{r[0], 0.0};
    double budget = tau * std::fabs(f[0]);
    std::printf("[ex3] exp(x)+exp(y)+0.5xy: c20=%.3f c11=%.3f c02=%.3f  reach=(%.4f,%.4f)\n",
                f[3], f[4], f[5], r[0], r[1]);
    std::printf("      single-axis err/budget at (r_x,0)=%.2f ;  BOX-CORNER (r_x,r_y)=%.2f"
                "  -> corner overshoots the budget\n",
                std::fabs(val::truncation_error(f, axis)) / budget,
                std::fabs(val::truncation_error(f, corner)) / budget);
    auto g = val::error_sensitivity(f, corner);
    std::printf("      blame at corner: alpha=%.4f beta=%.4f (h_i g_i)\n", corner[0] * g[0],
                corner[1] * g[1]);
    write_params("ex3", tau, f[0], r[0], r[1]);
    write_grid("ex3.csv", f, tau, 0.6,
               [](double hx, double hy) { return std::exp(hx) + std::exp(hy) + 0.5 * hx * hy; });
}

// ---- Example 4: a linear direction is free (infinite reach) ----------------
static void example4()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = 5.0 + 3.0 * x + y * y;  // linear in x, quadratic in y
    double const tau = 0.05;
    auto r = val::validity_radius(f, tau);
    std::printf("[ex4] 5+3x+y^2: reach_x=%.4g (inf -> x is free), reach_y=%.4f"
                "  (hand sqrt(5*tau)=%.4f)\n",
                r[0], r[1], std::sqrt(5 * tau));
    write_params("ex4", tau, f[0], r[0], r[1]);
    write_grid("ex4.csv", f, tau, 0.8, [](double hx, double hy) { return 5.0 + 3.0 * hx + hy * hy; });
}

// ---- Example 5: pure quadratic -- the order-2 estimate is EXACT -------------
static void example5()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = 1.0 + 2.0 * x - y + x * x + (x * y) + y * y;  // polynomial, no order-3+
    double const tau = 0.05;
    // residual real-est should be ~machine epsilon everywhere.
    double maxresid = 0.0;
    for (int iy = -50; iy <= 50; ++iy)
        for (int ix = -50; ix <= 50; ++ix) {
            double hx = 0.01 * ix, hy = 0.01 * iy;
            std::array<double, 2> step{hx, hy};
            double real = (1 + 2 * hx - hy + hx * hx + hx * hy + hy * hy) - val::evaluate(f, step);
            maxresid = std::fmax(maxresid, std::fabs(real - val::truncation_error(f, step)));
        }
    std::printf("[ex5] 1+2x-y+x^2+xy+y^2: max |real_err - est_err| = %.3e  (exact, ~machine eps)\n",
                maxresid);
    {
        auto r = val::validity_radius(f, tau);
        write_params("ex5", tau, f[0], r[0], r[1]);
    }
    write_grid("ex5.csv", f, tau, 0.6, [](double hx, double hy) {
        return 1 + 2 * hx - hy + hx * hx + hx * hy + hy * hy;
    });
}

int main()
{
    ::mkdir("data", 0755);
    { std::FILE* fp = std::fopen("data/params.csv", "w");
      std::fprintf(fp, "name,tau,f0,rx,ry\n"); std::fclose(fp); }
    example1();
    example2();
    example3();
    example4();
    example5();
    std::printf("wrote data/*.csv\n");
    return 0;
}
