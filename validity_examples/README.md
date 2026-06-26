# Validity examples — analytic intuition-builders

Nine tiny, hand-checkable examples for `otinum/validity.hpp`. Each builds an OTI
jet of a **known** function and exercises `evaluate` / `truncation_error` /
`is_trusted` / `validity_radius` / `error_sensitivity`, so you can see exactly what
the trust analysis does on cases where the answer is obvious.

```sh
c++ -std=c++17 -I ../include validity_examples.cpp -o validity_examples
./validity_examples            # prints hand-checks, writes data/*.csv
python plot_validity_examples.py   # writes figures/ex1..ex9.png
```

| Ex | Function | Concept | Hand-check |
|----|----------|---------|-----------|
| 1 | `exp(kx)` @0, `k∈{0.5,1,2}` | reach = where error crosses the budget; swept over curvature | reach `= √(2τ)/k` |
| 2 | `1/(1-x)` expanded toward its pole | a **nearby singularity collapses the reach** (and the error goes asymmetric) | reach `= √τ·(1−x₀)` |
| 3 | `1+2x−y+x²+xy+y²` | pure quadratic ⇒ **order-2 estimate is exact** (trust ellipse = exact level set) | residual `~1e-16` |
| 4 | `exp(x)+exp(y)+0.5xy` | **coupling**: tilted ellipse, box corners overshoot | corner error `= 3×` budget; blame splits 50/50 |
| 5 | `exp(x)` as `⟨1,3⟩` | the **`model_order` trade-off**: certify linear vs quadratic from one jet | reach `0.316 → 0.669` |
| 6 | `2+sin(x)` | a **vanishing leading term fools the estimate** (order-2 ≡ 0 ⇒ false ∞ reach) | naive reach `→ ∞`; honest `(6·budget)^⅓ = 0.843` |
| 7 | `exp(x)+exp(y)+0.5xy` | **`error_sensitivity` as a control step**: descend `−sign(E)∇E` back under budget | trusted in 3 steps |
| 8 | `1+x²+100y²` | **anisotropy**: a highly eccentric trust sliver | reachₓ/reach_y `= 10` |
| 9 | `1+x²−y²` | **saddle**: trust opens along the diagonals; per-axis reach can't express it | `E=0` on `hₓ=±h_y` (and at the box corners) |

## Example 1 in detail

The certified model is `otinum<1,2>` with `model_order = N-1 = 1`: the **linear**
surrogate `1+kh` is what you trust, and the order-2 term `½k²h²` is spent as the
truncation-error estimate. Note the horizontal axis is the **step `h`** — the
displacement away from the expansion point `x₀=0` — *not* the OTI variable `x`.

The **budget** is `τ·|f₀|`, an absolute error tolerance: here `τ=0.05` is a relative
fraction and `f₀ = exp(0) = 1`, so the budget is `0.05`. The **reach** `r` solves
`½k²r² = budget`, giving `r = √(2τ|f₀|)/k = √(2τ)/k`. The two **green** verticals on
the right panel mark `±r`: exactly where the `½k²h²` estimate crosses the budget.

The curvature knob `k` is swept to show the same picture at three bends:
`figures/ex1_lo.png` (k=0.5, gentle, large reach), `figures/ex1.png` (k=1, baseline),
`figures/ex1_hi.png` (k=2, sharp, small reach), and `figures/ex1_curvature.png`
overlays all three to show `reach ∝ 1/k` directly — doubling the curvature halves the
trusted step.

## Example 2 in detail

Same `otinum<1,2>` linear-surrogate format as example 1, but now the function
`1/(1-x)` has a **pole at x=1** and we expand at `x₀ ∈ {0, 0.5, 0.8}`, marching toward
it. With `d = 1−x₀` the coefficients are `[1/d, 1/d², 1/d³]`, so the budget is `τ/d`
and the reach is `r = √(budget/c₂) = √τ·d` — it **collapses linearly** with the
distance to the pole (`figures/ex2_far/mid/near.png`, and `figures/ex2_collapse.png`
overlays all three in *relative* error so they share one budget line).

The new lesson over example 1 is **asymmetry**: the pole sits on the `+h` side, so the
real error blows up much faster there than the *symmetric* quadratic estimate `|c₂|h²`
predicts. The certified reach is therefore mildly **optimistic toward the pole** and
conservative away from it — visible on the right panel where the black real-error curve
crosses the budget just inside the `+r` green line but outside the `−r` line.

## Examples 3 & 4 in detail (the 2D pair)

Both are `otinum<2,2>` over a 2D step `(hₓ, h_y)`. In the figures the **cyan** line is
the estimated trust region (`is_trusted`) and the **green dashed** line is the real
budget level set `|E| = τ|f₀|`; they coincide wherever the surrogate is calibrated.
Note `f₀ ≠ 1` now matters — the subtitle prints `f₀` and the resulting budget, so the
`|f₀|` factor that was invisible in example 1 (`f₀=1`) is explicit here.

- **Example 3 — `1+2x−y+x²+xy+y²` (exact).** A degree-2 polynomial lives entirely
  inside the certified model order, so there is *no* real truncation error: the order-2
  estimate matches the real error to machine epsilon (`residual ~1e-16`) and cyan lies
  exactly on green. This subsumes the old "a linear direction is free" example — a
  direction the function is exact in is just the degenerate `reach = ∞` case.
- **Example 4 — `exp(x)+exp(y)+0.5xy` (coupling).** The `xy` cross term tilts the trust
  ellipse off-axis. The **white dotted box** is the per-axis reach `(rₓ, r_y)`; its
  *corners* poke outside the trust region (corner error `≈ 3×` budget), the concrete
  "box ≠ safe, combine axes as an ellipse" lesson. `error_sensitivity` at the corner
  splits the blame 50/50 between the two variables.

## Examples 5 & 6 in detail (the `model_order` contract)

These two exercise the part `validity.hpp` calls "THE thing to get right": you certify
an order-`model_order` surrogate, and its leading truncation error is the
order-`(model_order+1)` band, so the jet must be computed at least one order higher.

- **Example 5 — the trade-off.** One `otinum<1,3>` jet of `exp(x)`, certified two ways
  via the `model_order` argument to `evaluate`/`truncation_error`/`validity_radius`:
  `model_order=1` trusts `1+h` (order-2 = error), `model_order=2` trusts `1+h+h²/2`
  (order-3 = error). Trusting the higher-order model grows the reach
  `√(2·budget) → (6·budget)^⅓`, i.e. `0.316 → 0.669`. Same data, more trust — at the
  cost of carrying one more order.
- **Example 6 — the failure mode.** `2+sin(x)` at 0 is *odd*, so the order-2 band is
  identically zero. A naive `otinum<1,2>` linear certification therefore inspects an
  all-zero error band, reports an effectively **infinite reach**, and `is_trusted`
  returns true for every `h` — even though the real error is **cubic** (`≈|h³/6|`).
  Computing one extra order (`otinum<1,3>`, certify `model_order=2`) exposes the cubic
  band and restores an honest reach `(6·budget)^⅓ = 0.843`. Moral: size the jet so the
  leading *non-zero* dropped term is the one the estimate inspects. (The `+2` keeps
  `f₀ ≠ 0` so the relative budget `τ|f₀|` is well defined.)

## Examples 7–9 in detail (control, anisotropy, the saddle limitation)

- **Example 7 — `error_sensitivity` as a control step.** Starting from an out-of-trust
  step, `error_sensitivity` (the gradient `∇E`) gives the steepest-error direction;
  iterating `h ← h − α·sign(E)·∇E` walks the step back across the `is_trusted` boundary
  in 3 iterations. This is the "attribute blame, then correct" loop the device-callable
  header is built for — diagnosis turned into action.
- **Example 8 — anisotropy.** `1+x²+100y²`: the `y` axis is 100× stiffer, so
  `reach_y` is 10× smaller than `reachₓ` and the trust region is a thin horizontal
  sliver. A single isotropic "trust radius" would either bust tolerance in `y` or waste
  the loose `x` axis — the reaches are genuinely per-axis semi-axes.
- **Example 9 — the saddle limitation.** `1+x²−y²`: the order-2 error `hₓ²−h_y²`
  **vanishes along the diagonals** `hₓ=±h_y`, so the true trust region is an open
  X-shaped band with infinite diagonal reach. `validity_radius` is axis-aligned and can
  only report a finite `(0.224, 0.224)` — it cannot express the open directions. Note
  the contrast with example 4: here the per-axis box *corners* land on the diagonal and
  are exactly **safe** (`E=0`), the opposite of the coupling case.
