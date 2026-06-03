# cpp_oti_lib Benchmarks

## Elementary Scaling

`elementary_scaling.cpp` compares the runtime of ordinary `double` evaluations
against first order OTI evaluations for dimensions `1, 2, 5, 10, 20, 30`.

The benchmark studies one case:

- `first_order_simple`: first order OTI derivatives for a polynomial expression
  using only addition and multiplication.

Each dimension is timed over multiple independent trials. The reported
`ratio` is computed from the median OTI time per evaluation divided by the
median real time per evaluation.

Build and run from the repository root:

```bash
c++ -std=c++17 -O3 -I include benchmarks/elementary_scaling.cpp -o /tmp/otinum_elementary_scaling
/tmp/otinum_elementary_scaling
```

The benchmark writes:

```text
benchmarks/results/elementary_scaling.csv
```

Generate PDF figures with:

```bash
python3 python_examples/benchmark_scaling.py
```

Figures are written to:

```text
python_examples/figures/benchmark_scaling/
```
