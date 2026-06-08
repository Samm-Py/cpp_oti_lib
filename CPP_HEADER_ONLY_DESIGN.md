# Header-Only C++ Port of `otilib` Static OTI Numbers — Design Document

**Status:** design notes for the current header-only implementation, with
some future-work sections still marked as sketches.
**Scope:** scalar OTI numbers only (`onumm<M>n<N>` family). Arrays, Gauss
quadrature, FEM, sparse/semisparse variants are explicitly out of scope for
the first pass and can be layered on later.
**Target:** a single `#include "otinum.hpp"` (or a small tree of headers)
providing a `template<int M, int N> class otinum` with natural operator
overloading, no build system, no linking to the existing C library.

---

## 1. What the library computes

An `onumm<M>n<N>` number is a **truncated multivariate Taylor polynomial**
in `M` formal infinitesimal variables `e_1, …, e_M`, truncated at total
order `N`. Each number stores one coefficient per monomial

```
e_1^{a_1} · e_2^{a_2} · … · e_M^{a_M}     with   |a| = a_1 + … + a_M ≤ N
```

The number of stored coefficients is

```
K(M, N) = C(M + N, N)      (binomial)
```

Examples: `K(2,2)=6`, `K(3,3)=20`, `K(10,2)=66`, `K(15,4)=3876`.

The existing C code stores these as named struct fields (`r, e1, e2, e11,
e12, e22, …`) with fully unrolled arithmetic per `(M,N)` pair. Our C++
version replaces all of that with a single templated class backed by
`std::array<double, K(M,N)>` and loop-based kernels.

Semantically identical to: `TaylorSeries.jl` (multivariate), Griewank/Walther
"forward mode to order N", `autodiff::HigherOrderDual`. The OTI phrasing
("order-truncated imaginary") is the same math dressed differently — the
`e_i` play the role of nilpotent generators with `e_i^{N+1} = 0`.

---

## 2. Public API sketch

```cpp
namespace oti {

template <int M, int N>
class otinum {
public:
    static constexpr int nvars   = M;
    static constexpr int order   = N;
    static constexpr int ncoeffs = detail::binom(M + N, N);

    // --- construction ---
    constexpr otinum() = default;                      // zero
    constexpr otinum(double r);                        // real lift
    static otinum variable(int i, double value = 0.0); // e_i with real part
    static otinum from_coeffs(std::array<double, ncoeffs> const&);

    // --- access ---
    double  real() const;                              // coeff of 1
    double  operator[](int flat_index) const;          // raw coeff access
    double& operator[](int flat_index);
    double  coeff(std::array<int, M> const& alpha) const; // ∂^α f / α! (normalized coeff)
    double  deriv(std::array<int, M> const& alpha) const; // compatibility alias for coeff()
    double  partial(std::array<int, M> const& alpha) const; // α! · coeff
    std::array<double, ncoeffs> const& data() const;

    // --- arithmetic (member-op form; free-function overloads below) ---
    otinum& operator+=(otinum const&);
    otinum& operator-=(otinum const&);
    otinum& operator*=(otinum const&);
    otinum& operator/=(otinum const&);
    otinum& operator+=(double);
    otinum& operator-=(double);
    otinum& operator*=(double);
    otinum& operator/=(double);

private:
    std::array<double, ncoeffs> c_{};  // zero-initialised
};

// --- free operators: all mix-and-match between otinum and double ---
template<int M,int N> otinum<M,N> operator+(otinum<M,N>, otinum<M,N> const&);
template<int M,int N> otinum<M,N> operator-(otinum<M,N>, otinum<M,N> const&);
template<int M,int N> otinum<M,N> operator*(otinum<M,N> const&, otinum<M,N> const&);
template<int M,int N> otinum<M,N> operator/(otinum<M,N> const&, otinum<M,N> const&);
template<int M,int N> otinum<M,N> operator-(otinum<M,N> const&);           // unary
// + symmetrical overloads with double on either side

// --- transcendentals (ADL-findable, match <cmath> names) ---
template<int M,int N> otinum<M,N> exp  (otinum<M,N> const&);
template<int M,int N> otinum<M,N> log  (otinum<M,N> const&);
template<int M,int N> otinum<M,N> log10(otinum<M,N> const&);
template<int M,int N> otinum<M,N> log_base(otinum<M,N> const&, double base);
template<int M,int N> otinum<M,N> pow  (otinum<M,N> const&, double e);
template<int M,int N> otinum<M,N> pow  (otinum<M,N> const&, otinum<M,N> const&);
template<int M,int N> otinum<M,N> sqrt (otinum<M,N> const&);
template<int M,int N> otinum<M,N> cbrt (otinum<M,N> const&);
template<int M,int N> otinum<M,N> sin  (otinum<M,N> const&);
template<int M,int N> otinum<M,N> cos  (otinum<M,N> const&);
template<int M,int N> otinum<M,N> tan  (otinum<M,N> const&);
template<int M,int N> otinum<M,N> asin (otinum<M,N> const&);
template<int M,int N> otinum<M,N> acos (otinum<M,N> const&);
template<int M,int N> otinum<M,N> atan (otinum<M,N> const&);
template<int M,int N> otinum<M,N> sinh (otinum<M,N> const&);
template<int M,int N> otinum<M,N> cosh (otinum<M,N> const&);
template<int M,int N> otinum<M,N> tanh (otinum<M,N> const&);
template<int M,int N> otinum<M,N> asinh(otinum<M,N> const&);
template<int M,int N> otinum<M,N> acosh(otinum<M,N> const&);
template<int M,int N> otinum<M,N> atanh(otinum<M,N> const&);
template<int M,int N> otinum<M,N> abs  (otinum<M,N> const&);

// --- truncated variants matching the C API (optional but cheap) ---
template<int M,int N> otinum<M,N> trunc_mul(otinum<M,N> const&, otinum<M,N> const&, int max_order);
template<int M,int N> otinum<M,N> trunc_add(otinum<M,N> const&, otinum<M,N> const&, int max_order);
template<int M,int N> otinum<M,N> gem (otinum<M,N> const& a, otinum<M,N> const& b, otinum<M,N> const& c); // a*b + c

} // namespace oti
```

Usage:

```cpp
using T = oti::otinum<2, 2>;
T x = T::variable(0, 1.5);   // 1.5 + e_1
T y = T::variable(1, 0.3);   // 0.3 + e_2
T f = sin(x*y) + exp(x);
double d2f_dxdy = f.partial({1, 1});   // ∂²f / ∂x ∂y
```

---

## 3. Coefficient layout and the multi-index table

The central design choice. All arithmetic reduces to operations on a flat
`std::array<double, K(M,N)>`. We need a bijection

```
index  ↔  multi-index α = (a_1, …, a_M),  |α| ≤ N
```

### 3.1 Ordering

Use **graded reverse lexicographic** order: monomials grouped by total
order `|α|`, within each order ordered lexicographically on `(a_1, …, a_M)`.
This makes "truncate to order k" a prefix operation (coeffs `[0, K(M,k))`),
which matches how the C code does truncated multiply and order-bounded
sum / GEM.

For `(M,N) = (2,2)` the order is:
```
0: (0,0)                       → flat index 0   (real part)
1: (1,0) (0,1)                 → 1, 2
2: (2,0) (1,1) (0,2)           → 3, 4, 5
```
matching the C struct layout `r, e1, e2, e11, e12, e22`. Good — the C
reference is a direct sanity check.

### 3.2 Precomputed tables (all `constexpr`)

Built once per `(M,N)` instantiation at compile time:

| table | shape | purpose |
|---|---|---|
| `idx_to_alpha[k]`          | `K × M` (`int8_t`) | flat index → exponent vector |
| `alpha_to_idx`             | `(N+1)^M` or hash  | inverse (see 3.3) |
| `order_of[k]`              | `K` (`int8_t`)     | `\|α\|` per flat index |
| `order_offset[o]`          | `N+2`              | first flat index of order `o` |
| `factorial_alpha[k]`       | `K` (`double`)     | `α! = a_1! … a_M!` for `deriv ↔ partial` |
| `conv_table`               | list of `(i, j, k)` triples | see §4.2 |

The tables live in a `detail::Tables<M,N>` with all members
`static constexpr` (C++20; use `inline constexpr` arrays or an `inline`
static function returning a reference for C++17).

### 3.3 `alpha_to_idx`

Two options:
- **Dense** `(N+1)^M` lookup. Cheap when `M·log2(N+1) ≤ ~20`. Blows up for
  `M=50, N=2` (≈7.2·10^23) — unusable at the large end.
- **Ranking function.** Closed-form graded-lex rank of `(a_1, …, a_M)` from
  cumulative binomials. Constant memory, `O(M)` per rank. This is the safe
  default; the dense table is an optimisation for small `(M,N)`.

Decision: implement the rank function. It is the only thing that needs to
be correct for large modules.

---

## 4. Arithmetic kernels

### 4.1 Linear ops

`+`, `-`, unary `-`, scalar `*`/`/` are element-wise over the flat array —
trivial `for` loops the compiler will vectorise. No multi-index logic
needed. Mixing with `double` operates only on `c_[0]` (the real
coefficient).

### 4.2 Multiplication — the truncated convolution

For `a · b` in the polynomial ring modulo `(|α| > N)`:

```
c[γ] = Σ_{α + β = γ,   |α|+|β| = |γ| ≤ N}  a[α] · b[β]
```

Build once, at compile time, a flat list of triples `(i, j, k)` such that
`alpha(i) + alpha(j) = alpha(k)` and `order[k] ≤ N`. The multiplication
kernel is then:

```cpp
for (auto [i, j, k] : Tables<M,N>::conv_table)
    c[k] += a[i] * b[j];
```

Size of `conv_table`: the number of triples is exactly `Σ_{|γ|≤N} C(|γ|+M-1, M-1)^2`
(bounded, and the compiler can place it in rodata). For `(15,4)` it is
large but finite; we can chunk or generate on first call if binary size is
a concern.

Two optimisations that are worth the complexity later but **not** in v1:
- Group triples by `k` to enable FMA-friendly inner loops.
- Exploit symmetry: for `γ = α + β`, only store `α ≤ β` and double.

### 4.3 Truncated multiply (`trunc_mul`, `gem` with order cutoff)

Same kernel, filter triples on `max(order[i], order[j], order[k]) ≤
user_order`. Either walk the full table and skip, or generate a family of
tables indexed by cutoff. Start with "walk and skip" — simpler, still fast.

### 4.4 Division

`a / b = a · (1/b)`. Compute `1/b` via the univariate Taylor recurrence on
the scalar function `g(x) = 1/x` composed with `b` — see §5. Equivalently
and slightly cheaper: use the fact that `1/b = (1/b.real()) · 1/(1 + h)`
where `h = (b - b.real()) / b.real()` is a pure nilpotent (no real part,
so `h^{N+1} = 0`), and expand `1/(1+h) = Σ_{k=0}^{N} (-h)^k` via `N`
truncated multiplications. Clean, easy to verify, `O(N · K log K)` in
practice. Same trick applies to `sqrt`, `log`, `exp`, `pow` and is the
foundation of §5.

---

## 5. Transcendental functions — univariate Taylor composition

**Key fact.** Any `f(b)` for smooth `f: ℝ → ℝ` applied to an OTI number `b`
depends only on:
1. the real part `b_0 = b.real()`, evaluated through the standard
   `<cmath>` function, and
2. the nilpotent part `h = b - b_0`, which satisfies `h^{N+1} = 0`.

Then
```
f(b) = Σ_{k=0}^{N} f^{(k)}(b_0) / k! · h^k
```
is a finite sum — exactly `N` truncated multiplications plus scalar
Taylor-coefficient evaluation for the specific function `f`. This single
recipe handles **all** transcendentals in the C library.

### 5.1 The scalar Taylor coefficients

For each supported `f`, provide a function that returns
`{f(x_0), f'(x_0), …, f^{(N)}(x_0) / N!}` (or equivalently the unnormalised
derivatives plus division by factorials). Either:

- **Direct closed form** (cheap, preferred where it exists):
  - `exp`:  `t_k = exp(x_0) / k!`
  - `log`:  `t_0 = log(x_0);  t_k = (-1)^{k+1} / (k · x_0^k)`
  - `pow(·, p)`: `t_k = C(p, k) · x_0^{p-k}`  (generalised binomial)
  - `sin`, `cos`: `t_k = sin/cos(x_0 + kπ/2) / k!`
  - `sinh`, `cosh`: similar.

- **ODE-based Taylor recurrence** (systematic, covers the rest):
  Each standard function satisfies a simple ODE; differentiating it gives a
  recurrence that produces `t_{k+1}` from `t_k, …, t_0` in `O(k)` work.
  Examples:
  - `y = tan(x)`: `y' = 1 + y^2`
  - `y = atan(x)`: `(1+x^2) y' = 1`
  - `y = sqrt(x)`: `2 y y' = 1`
  - `y = asin(x)`: `sqrt(1-x^2) y' = 1`

  A single generic routine `taylor_from_ode(x_0, ode_spec)` can produce
  coefficients for all inverse-trig / inverse-hyperbolic cases, matching
  what the C library does internally.

### 5.2 The nilpotent expansion kernel

```cpp
template<int M,int N>
otinum<M,N> apply_scalar(std::array<double, N+1> const& t, otinum<M,N> const& b)
{
    otinum<M,N> h = b;  h[0] = 0.0;              // nilpotent part
    otinum<M,N> hk(1.0);                         // h^0
    otinum<M,N> out(t[0]);
    for (int k = 1; k <= N; ++k) {
        hk = trunc_mul(hk, h, N);                // h^k, but upper orders zero anyway
        out += t[k] * hk;
    }
    return out;
}
```

Every `f` is then a 2-liner: compute `t[]`, call `apply_scalar`. This
replaces the ~2000 lines of hand-written per-`(M,N)` transcendental code.

### 5.3 `pow(a, b)` with both operands OTI

`pow(a, b) = exp(b · log(a))`. Composition of the above. Same recipe.

### 5.4 `abs`

`abs` is not smooth at zero. Match the C library's behaviour: `abs(x) =
sign(x.real()) · x`. Document this as "derivative at zero is undefined;
result at `x.real() == 0` follows the C reference."

---

## 6. Testing plan

The existing C library is the oracle. Build a small test harness that:

1. Links the existing `libotilib` for a handful of `(M,N)` pairs that
   actually exist as hand-written modules: at minimum `{(1,1), (2,2),
   (3,3), (4,4), (10,2), (15,4)}` to cover both tiny and large cases.
2. For each op (`+ - * / neg`, all transcendentals, `pow`, `trunc_mul`,
   `gem`):
   - generate random inputs (real parts away from singularities),
   - compute via C and via C++,
   - assert coefficient-wise agreement to ~1e-12 relative.
3. Property tests:
   - `exp(log(x)) == x` (within tol), for `x.real() > 0`;
   - `sin^2 + cos^2 == 1`;
   - `(a + b) · c == a·c + b·c` (distributivity);
   - derivative values against finite differences on `f: ℝ^M → ℝ` for small
     `M`.
4. `static_assert` the compile-time tables for known small cases — e.g.
   `Tables<2,2>::idx_to_alpha` matches `{(0,0),(1,0),(0,1),(2,0),(1,1),
   (0,2)}`.

---

## 7. File layout

Header-only, split for readability and compile speed (only the headers you
include get instantiated):

```
include/otinum/
    otinum.hpp              // umbrella — pulls everything
    detail/
        binom.hpp           // constexpr binomial, factorial
        multi_index.hpp     // Tables<M,N>, ranking function
        convolution.hpp     // conv_table builder, trunc_mul kernel
    core.hpp                // class otinum + linear ops + mul/div
    functions.hpp           // all <cmath>-style overloads
    taylor.hpp              // apply_scalar + per-function t[] generators
```

No `.cpp`. No CMake required to consume; a single `-I include/` is enough.
C++17 minimum (use `if constexpr`, fold expressions, `std::array`,
`constexpr` enough of the tables). C++20 preferred (cleaner `constexpr`
containers, concepts for the numeric overload set).

---

## 8. Performance notes and non-goals

- **Compile time / binary size** scale with `K(M,N)^2` for the convolution
  table. `(M,N) = (15,4)` gives `K = 3876`, conv table triples in the tens
  of millions. For the large modules we should either (a) generate the
  table lazily on first use into a `static` heap buffer rather than
  `constexpr`, or (b) skip the table entirely and loop `i,j` recomputing
  `γ = α_i + α_j` + rank. Pick (b) as the default and offer (a) behind a
  flag. This is a deliberate departure from the "fully unrolled" C style;
  with `-O3` the loop version is within a small factor and compiles in
  seconds instead of minutes.
- **SIMD / BLAS-style GEMM over OTI**: out of scope. Array and matrix ops
  live in the existing C code; a later phase can add `otimat<M,N>` using
  `std::mdspan` or a tiny Eigen-style wrapper.
- **Thread safety**: the class is a value type with no shared state;
  trivially thread-safe by construction. The precomputed tables are
  read-only.
- **Allocator awareness / custom coefficient types**: not in v1. Everything
  is `double`. A second template parameter `Coeff = double` can be added
  later without API break.
- **Exceptions**: none thrown. Domain errors (e.g. `log` of a number with
  non-positive real part) propagate the `<cmath>` behaviour (NaN).

---

## 9. Effort estimate (recap from the conversation)

| Piece | Estimate |
|---|---|
| `Tables<M,N>` + rank function + `conv_table` | 1 day |
| Class skeleton, linear ops, `*`, `/` | 1 day |
| `apply_scalar` + scalar Taylor coefficient generators for all functions | 1–2 days |
| Test harness against C reference | 1 day |
| Docs / examples | 0.5 day |
| **Total for usable scalar header-only lib** | **~1 week** |

Array / Gauss / FEM / sparse would each add comparable effort and are
deferred.

---

## 10. Open questions to revisit

1. Do we want bit-identical agreement with the C reference, or only
   numerical agreement? The C code has a specific operation ordering that
   affects the last few ULPs; matching it exactly constrains the kernel
   implementation.
2. Do we need `constexpr` evaluation of OTI arithmetic (useful for
   compile-time Jacobians), or is runtime enough? `constexpr` pushes the
   minimum C++ version toward 20/23.
3. Is `(M,N) = (50, 2)` or `(15, 4)` actually exercised in user code, or
   are the large hand-written modules vestigial? If they are unused we can
   cap at, say, `K(M,N) ≤ 1000` and punt on the table-size problem.
4. Do we want an `otinum_dynamic` runtime-`(M,N)` variant too, or is the
   templated compile-time form sufficient? The existing C library is
   compile-time-per-module; mirroring that keeps scope tight.
