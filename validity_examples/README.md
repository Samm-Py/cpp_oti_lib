# Validity examples — analytic intuition-builders

Five tiny, hand-checkable examples for `otinum/validity.hpp`. Each builds an OTI
jet of a **known** function and exercises `evaluate` / `truncation_error` /
`is_trusted` / `validity_radius` / `error_sensitivity`, so you can see exactly what
the trust analysis does on cases where the answer is obvious.

```sh
c++ -std=c++17 -I ../include validity_examples.cpp -o validity_examples
./validity_examples            # prints hand-checks, writes data/*.csv
python plot_validity_examples.py   # writes figures/ex1..ex5.png
```

| Ex | Function | Concept | Hand-check |
|----|----------|---------|-----------|
| 1 | `exp(x)` @0 | reach = where error crosses the budget | reach `= √(2τ)` |
| 2 | `exp(kx)`; `1/(1-x)` near pole | reach responds to curvature / nonlinearity | reach `∝ 1/k`; collapses ∝ distance to pole |
| 3 | `exp(x)+exp(y)+0.5xy` | **coupling**: tilted ellipse, box corners overshoot | corner error `= 3×` budget; blame splits 50/50 |
| 4 | `5+3x+y²` | a **linear direction is free** | `reach_x = ∞`, `reach_y = √(5τ)` |
| 5 | `1+2x−y+x²+xy+y²` | pure quadratic ⇒ **order-2 estimate is exact** | residual `~1e-16` |

In the 2D figures the **cyan** line is the estimated trust region (`is_trusted`),
the **green dashed** line is the real budget level set — they coincide where the
surrogate is calibrated. In example 3 the **white box** is the per-axis reach; its
corners poke outside the trust region, the concrete "box ≠ safe" lesson. In example
5 cyan and green lie exactly on top of each other (the estimate is exact).
