# mpi_oti_reduce — global reduction with a custom MPI_Op over jets

A distributed **quantity of interest** that carries its own gradient and Hessian.
Each rank owns a block of an `N×N` grid and accumulates a partial sum of
`f(x, y; a, b) = sin(a·x)·exp(b·y)`, where the design parameters `a, b` are
**seeded as OTI variables**. One `MPI_Allreduce` with a custom jet-sum operator
folds the per-rank partial-sum jets into the global QoI — value, gradient, and
Hessian w.r.t. `(a, b)` — on every rank, from a single reduction. That's the
gradient *and* Hessian of a global objective over a distributed field, with no
adjoint and no extra communication.

The jet is `oti::otinum<2, 2, double>` (value + 2 first + 3 second derivatives).

## The custom MPI_Op (from the header)

MPI has no built-in way to combine an `otinum`, so the reduction needs a user
`MPI_Op`. You don't hand-roll it — `otinum/mpi.hpp` provides it:

```cpp
MPI_Datatype MPI_OTINUM  = oti::mpi::make_datatype<Jet>();
MPI_Op       MPI_OTI_SUM = oti::mpi::make_sum_op<Jet>();   // sums jets
MPI_Allreduce(&local, &global, 1, MPI_OTINUM, MPI_OTI_SUM, MPI_COMM_WORLD);
oti::mpi::free_op(MPI_OTI_SUM);
```

`make_sum_op` registers a callback whose body is just `otinum::operator+=`
applied across MPI's element buffer — the explicit per-element loop is only there
because MPI's reduction callback is type-erased (`void*`) and may batch many
elements per call; it is *not* re-defining jet addition. For a **sum**, jet
addition is coefficient-wise, so this equals `MPI_SUM` over the raw coefficients —
you could reduce the underlying doubles instead. The custom op is the *general*
mechanism: required the moment the combine is not coefficient-wise (e.g. reducing
a **product** of jets, where `operator*` is a convolution), and it keeps the
reduction in `MPI_OTINUM` units.

The header ships the same family for the other combines, all built on a generic
`make_reduce_op<T, Op>()`:

| Builder | Combine | Use |
|---------|---------|-----|
| `make_sum_op<T>()` | `a + b` | additive QoI (this example) |
| `make_prod_op<T>()` | `a * b` (convolution) | multiplicative QoI |
| `make_max_op<T>()` / `make_min_op<T>()` | jet with larger/smaller value | value of an extremum + its sensitivity |

`make_max_op`/`make_min_op` give the sensitivity *at* the argmax/argmin (valid
where the extremum is unique and interior). For anything else, pass your own
functor to `make_reduce_op`. `test_reduce_ops.cpp` checks all four against a
serial recompute (`mpirun -np 4 ./test_reduce_ops`).

## Verification

Unlike the gather and halo examples, the result is **not bit-identical** across
rank counts: floating-point addition is not associative, so a different partition
sums in a different order. Two tolerance-based checks instead:

1. **Distributed vs serial** — the global jet against a single-process recompute,
   to a tight relative tolerance (`1e-10`). The difference is pure
   summation-order rounding (`~1e-14`; exactly `0` at `np=1`).
2. **Gradient vs finite differences** — `d/da`, `d/db` against centred FD on the
   parameters (`h=1e-6`), agreeing to `~1e-8`. This independently confirms the
   reduced derivatives are correct.

## Build & run

```sh
mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_reduce
mpirun -np 4 ./mpi_oti_reduce
```

Verified at `np = 1, 3, 4`. The program returns nonzero if either check fails, so
it works as a CI gate.
