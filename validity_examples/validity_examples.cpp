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

// ---- Example 1: 1D exp(k x) around 0 -- the hello world, swept over curvature -
// model_order = N-1 = 1, so the CERTIFIED surrogate is LINEAR (1 + k h) and the
// order-2 term (k^2/2) h^2 is the truncation-error estimate. The curvature knob k
// directly sets the bend: reach = sqrt(2 tau)/k, so doubling k halves the reach.
static void example1_one(char const* fname, double k, double tau)
{
    using J = oti::otinum<1, 2, double>;
    J const f = oti::exp(J::variable(0, 0.0) * k);  // coeffs [1, k, k^2/2]
    double const r = val::validity_radius(f, tau)[0];
    double const budget = tau * std::fabs(f[0]);
    std::printf("[ex1] exp(%.2f x) @0: c0=%.3f c1=%.3f c2=%.3f  budget=tau*|f0|=%.3f"
                "  reach(tau=%.2f)=%.4f  (hand sqrt(2 tau)/k=%.4f)\n",
                k, f[0], f[1], f[2], budget, tau, r, std::sqrt(2 * tau) / k);

    std::FILE* fp = open_csv(fname);
    // k and budget are constant per file but carried per-row so the plotter needs
    // no second source of truth for the curvature / budget it annotates.
    std::fprintf(fp, "h,truth,linear,real_err,est_err,trusted,reach,k,budget\n");
    for (int j = -100; j <= 100; ++j) {
        double h = 0.01 * j;
        std::array<double, 1> step{h};
        double lin = val::evaluate(f, step);
        double truth = std::exp(k * h);
        std::fprintf(fp, "%.4f,%.6g,%.6g,%.6g,%.6g,%d,%.6f,%.3f,%.6f\n", h, truth, lin,
                     truth - lin, val::truncation_error(f, step),
                     val::is_trusted(f, step, tau) ? 1 : 0, r, k, budget);
    }
    std::fclose(fp);
}

static void example1()
{
    double const tau = 0.05;
    example1_one("ex1_lo.csv", 0.5, tau);  // LOWER curvature  -> larger reach
    example1_one("ex1.csv", 1.0, tau);     // baseline hello-world (exp(x))
    example1_one("ex1_hi.csv", 2.0, tau);  // HIGHER curvature -> smaller reach
}

// ---- Example 2: a nearby POLE collapses the reach (1/(1-x) toward x=1) -------
// Same ex1 format, but the function has a singularity at x=1 and we expand at
// several x0 marching toward it. With d = 1-x0 the coeffs are [1/d, 1/d^2, 1/d^3],
// so budget = tau/d and reach r = sqrt(tau)*d shrinks linearly with the distance
// to the pole. Unlike exp, the REAL error is ASYMMETRIC -- it blows up toward the
// pole (+h) far faster than the symmetric quadratic estimate predicts, so the
// certified reach is mildly OPTIMISTIC on the pole side and conservative away.
static void example2_one(char const* fname, double x0, double tau)
{
    using J = oti::otinum<1, 2, double>;
    J const f = 1.0 / (1.0 - J::variable(0, x0));  // coeffs [1/d, 1/d^2, 1/d^3]
    double const d = 1.0 - x0;
    double const r = val::validity_radius(f, tau)[0];
    double const budget = tau * std::fabs(f[0]);
    std::printf("[ex2] 1/(1-x) @%.2f (dist d=%.2f): c0=%.3f c1=%.3f c2=%.3f"
                "  budget=%.4f reach=%.4f  (hand sqrt(tau)*d=%.4f)\n",
                x0, d, f[0], f[1], f[2], budget, r, std::sqrt(tau) * d);

    std::FILE* fp = open_csv(fname);
    std::fprintf(fp, "h,truth,linear,real_err,est_err,trusted,reach,x0,d,budget,f0\n");
    // Sweep h within +/-0.9 d (h measured as a fraction of the pole distance) so
    // the truth 1/(d-h) stays finite on both sides while the +h side nears the pole.
    for (int j = -90; j <= 90; ++j) {
        double h = 0.01 * j * d;
        std::array<double, 1> step{h};
        double lin = val::evaluate(f, step);
        double truth = 1.0 / (d - h);
        std::fprintf(fp, "%.5f,%.6g,%.6g,%.6g,%.6g,%d,%.6f,%.3f,%.4f,%.6f,%.6f\n", h, truth,
                     lin, truth - lin, val::truncation_error(f, step),
                     val::is_trusted(f, step, tau) ? 1 : 0, r, x0, d, budget, f[0]);
    }
    std::fclose(fp);
}

static void example2()
{
    double const tau = 0.05;
    example2_one("ex2_far.csv", 0.0, tau);   // far from pole  (d=1.0): large reach
    example2_one("ex2_mid.csv", 0.5, tau);   // approaching    (d=0.5)
    example2_one("ex2_near.csv", 0.8, tau);  // close to pole  (d=0.2): reach collapses
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

// ---- Example 3: pure quadratic -- the order-2 estimate is EXACT -------------
// A degree-2 polynomial lives entirely inside the model order, so the certified
// surrogate has NO real truncation error: the order-2 estimate matches the real
// error to machine epsilon and the trust ellipse IS the exact budget level set.
// This subsumes the old "a linear direction is free" example -- a direction the
// function is exact in is just the degenerate case of infinite reach.
static void example3()
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
    std::printf("[ex3] 1+2x-y+x^2+xy+y^2: max |real_err - est_err| = %.3e  (exact, ~machine eps)\n",
                maxresid);
    {
        auto r = val::validity_radius(f, tau);
        write_params("ex3", tau, f[0], r[0], r[1]);
    }
    write_grid("ex3.csv", f, tau, 0.6, [](double hx, double hy) {
        return 1 + 2 * hx - hy + hx * hx + hx * hy + hy * hy;
    });
}

// ---- Example 4: coupling -- the tilted ellipse and box-corner overshoot ----
static void example4()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = oti::exp(x) + oti::exp(y) + 0.5 * (x * y);  // coupled, definite Hessian
    double const tau = 0.05;
    auto r = val::validity_radius(f, tau);
    std::array<double, 2> corner{r[0], r[1]}, axis{r[0], 0.0};
    double budget = tau * std::fabs(f[0]);
    std::printf("[ex4] exp(x)+exp(y)+0.5xy: c20=%.3f c11=%.3f c02=%.3f  reach=(%.4f,%.4f)\n",
                f[3], f[4], f[5], r[0], r[1]);
    std::printf("      single-axis err/budget at (r_x,0)=%.2f ;  BOX-CORNER (r_x,r_y)=%.2f"
                "  -> corner overshoots the budget\n",
                std::fabs(val::truncation_error(f, axis)) / budget,
                std::fabs(val::truncation_error(f, corner)) / budget);
    auto g = val::error_sensitivity(f, corner);
    std::printf("      blame at corner: alpha=%.4f beta=%.4f (h_i g_i)\n", corner[0] * g[0],
                corner[1] * g[1]);
    write_params("ex4", tau, f[0], r[0], r[1]);
    write_grid("ex4.csv", f, tau, 0.6,
               [](double hx, double hy) { return std::exp(hx) + std::exp(hy) + 0.5 * hx * hy; });
}

// ---- Example 5: the model_order trade-off -- one jet, linear vs quadratic ----
// otinum<1,3> of exp(x): certify the SAME jet two ways. model_order=1 trusts the
// linear surrogate 1+h; model_order=2 trusts 1+h+h^2/2. The error functions fold
// in EVERY stored order above the model, so the linear reach (mo=1) accounts for
// the order-2 AND order-3 bands -> ~0.301, slightly under the leading-order-only
// sqrt(2 budget)=0.316; the quadratic reach (mo=2) has only the order-3 band, so
// it stays the closed form (6 budget)^(1/3)=0.669. Trusting the higher-order
// model still enlarges the reach. evaluate/trunc/radius all take model_order.
static void example5()
{
    using J = oti::otinum<1, 3, double>;
    J const f = oti::exp(J::variable(0, 0.0));  // coeffs [1, 1, 1/2, 1/6]
    double const tau = 0.05;
    double const budget = tau * std::fabs(f[0]);
    double const r1 = val::validity_radius(f, tau, 1)[0];
    double const r2 = val::validity_radius(f, tau, 2)[0];
    std::printf("[ex5] exp(x) <1,3>: reach(linear,mo=1)=%.4f (folds order-2+3; leading-order"
                " sqrt(2 budget)=%.4f); reach(quad,mo=2)=%.4f (hand (6 budget)^1/3=%.4f)\n",
                r1, std::sqrt(2 * budget), r2, std::cbrt(6 * budget));
    std::FILE* fp = open_csv("ex5.csv");
    std::fprintf(fp, "h,truth,surr1,surr2,real1,est1,real2,est2,reach1,reach2,budget\n");
    for (int j = -120; j <= 120; ++j) {
        double h = 0.01 * j;
        std::array<double, 1> step{h};
        double truth = std::exp(h);
        double s1 = val::evaluate(f, step, 1), s2 = val::evaluate(f, step, 2);
        std::fprintf(fp, "%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6f,%.6f,%.6f\n", h, truth,
                     s1, s2, truth - s1, val::truncation_error(f, step, 1), truth - s2,
                     val::truncation_error(f, step, 2), r1, r2, budget);
    }
    std::fclose(fp);
}

// ---- Example 6: a vanishing leading term fools the estimate (2 + sin x) ------
// At x0=0, sin has NO quadratic term (c2=0), so a linear certification's only error
// band (order 2) is identically zero: validity reports infinite reach and is_trusted
// is always true -- yet the real error is CUBIC. Computing one extra order
// (otinum<1,3>, certify model_order=2) exposes the cubic band and restores an honest
// finite reach. (The +2 keeps f0 nonzero so the relative budget tau*|f0| is defined.)
static void example6()
{
    using J2 = oti::otinum<1, 2, double>;  // naive: linear model, order-2 error band
    using J3 = oti::otinum<1, 3, double>;  // fixed: carries the cubic band too
    J2 const f2 = 2.0 + oti::sin(J2::variable(0, 0.0));  // [2, 1, 0]
    J3 const f3 = 2.0 + oti::sin(J3::variable(0, 0.0));  // [2, 1, 0, -1/6]
    double const tau = 0.05;
    double const budget = tau * 2.0;
    double const r_naive = val::validity_radius(f2, tau, 1)[0];  // -> inf (c2 == 0)
    double const r_fixed = val::validity_radius(f3, tau, 2)[0];  // finite, from cubic
    std::printf("[ex6] 2+sin(x): naive linear reach=%.4g (c2=0 -> INF, WRONG);"
                " quad(mo=2) reach=%.4f (hand (6 budget)^1/3=%.4f)\n",
                r_naive, r_fixed, std::cbrt(6 * budget));
    std::FILE* fp = open_csv("ex6.csv");
    std::fprintf(fp, "h,truth,surr,real_err,est_naive,est_fixed,trusted_naive,trusted_fixed,"
                     "reach_fixed,budget\n");
    for (int j = -150; j <= 150; ++j) {
        double h = 0.01 * j;
        std::array<double, 1> step{h};
        double truth = 2.0 + std::sin(h);
        double surr = val::evaluate(f2, step, 1);  // 2 + h
        std::fprintf(fp, "%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%.6f,%.6f\n", h, truth, surr,
                     truth - surr, val::truncation_error(f2, step, 1),       // == 0
                     val::truncation_error(f3, step, 2),                     // cubic
                     val::is_trusted(f2, step, tau, 1) ? 1 : 0,
                     val::is_trusted(f3, step, tau, 2) ? 1 : 0, r_fixed, budget);
    }
    std::fclose(fp);
}

// ---- Example 7: error_sensitivity as a corrective control step --------------
// Start at an out-of-trust step h; error_sensitivity gives g = grad E(h). Stepping
// DOWN the |error| gradient (h <- h - alpha*sign(E)*g) pulls the step back under
// budget in a few iterations -- the "attribute blame, then correct" control loop
// that the device-callable header is meant for.
static void example7()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = oti::exp(x) + oti::exp(y) + 0.5 * (x * y);  // coupled, definite Hessian
    double const tau = 0.05;
    double const budget = tau * std::fabs(f[0]);
    auto r = val::validity_radius(f, tau);
    write_params("ex7", tau, f[0], r[0], r[1]);
    write_grid("ex7.csv", f, tau, 0.6,
               [](double hx, double hy) { return std::exp(hx) + std::exp(hy) + 0.5 * hx * hy; });

    std::FILE* fp = open_csv("ex7_path.csv");
    std::fprintf(fp, "iter,hx,hy,err,trusted\n");
    std::array<double, 2> h{0.55, 0.45};  // start outside the trust region
    double const alpha = 0.18;
    std::printf("[ex7] correcting an out-of-trust step via error_sensitivity:\n");
    for (int it = 0; it < 30; ++it) {
        double E = val::truncation_error(f, h);
        bool tr = val::is_trusted(f, h, tau);
        std::fprintf(fp, "%d,%.5f,%.5f,%.6g,%d\n", it, h[0], h[1], E, tr ? 1 : 0);
        if (it < 3 || tr)
            std::printf("        it=%2d  h=(%.3f,%.3f)  |E|/budget=%.2f  %s\n", it, h[0], h[1],
                        std::fabs(E) / budget, tr ? "TRUSTED" : "");
        if (tr) break;
        auto g = val::error_sensitivity(f, h);
        double s = E >= 0 ? 1.0 : -1.0;
        h[0] -= alpha * s * g[0];
        h[1] -= alpha * s * g[1];
    }
    std::fclose(fp);
}

// ---- Example 8: anisotropy -- a highly eccentric trust ellipse --------------
// f = 1 + x^2 + 100 y^2: the y direction is 100x stiffer, so reach_y is 10x smaller
// than reach_x. The trust region is a thin sliver; a single isotropic radius would
// waste the loose x axis. Pure quadratic, so the estimate is exact.
static void example8()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = 1.0 + x * x + 100.0 * (y * y);
    double const tau = 0.05;
    auto r = val::validity_radius(f, tau);
    std::printf("[ex8] 1+x^2+100y^2: reach=(%.4f,%.4f) ratio=%.1f"
                "  (hand sqrt(tau)=%.4f, sqrt(tau/100)=%.4f)\n",
                r[0], r[1], r[0] / r[1], std::sqrt(tau), std::sqrt(tau / 100.0));
    write_params("ex8", tau, f[0], r[0], r[1]);
    write_grid("ex8.csv", f, tau, 0.3,
               [](double hx, double hy) { return 1.0 + hx * hx + 100.0 * hy * hy; });
}

// ---- Example 9: a saddle -- per-axis reach misses the open diagonals --------
// f = 1 + x^2 - y^2: the order-2 error hx^2 - hy^2 VANISHES along hx = +-hy, so the
// true trust region is an open X-shaped band along the diagonals. validity_radius is
// axis-aligned and reports a finite (0.224, 0.224) box it cannot express the infinite
// diagonal reach. Contrast example 4: here the box CORNERS are exactly safe (E=0).
static void example9()
{
    using J = oti::otinum<2, 2, double>;
    J const x = J::variable(0, 0.0), y = J::variable(1, 0.0);
    J const f = 1.0 + x * x - y * y;
    double const tau = 0.05;
    auto r = val::validity_radius(f, tau);
    std::array<double, 2> diag{0.5, 0.5}, corner{r[0], r[1]};
    std::printf("[ex9] 1+x^2-y^2 saddle: per-axis reach=(%.4f,%.4f);"
                " E on diagonal(0.5,0.5)=%.1e (==0); E at box corner=%.1e (==0)\n",
                r[0], r[1], val::truncation_error(f, diag), val::truncation_error(f, corner));
    write_params("ex9", tau, f[0], r[0], r[1]);
    write_grid("ex9.csv", f, tau, 0.4,
               [](double hx, double hy) { return 1.0 + hx * hx - hy * hy; });
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
    example6();
    example7();
    example8();
    example9();
    std::printf("wrote data/*.csv\n");
    return 0;
}
