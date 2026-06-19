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

## The custom MPI_Op

MPI has no built-in way to combine an `otinum`, so we register one:

```cpp
void jet_sum(void* in, void* inout, int* len, MPI_Datatype*) {
    const Jet* a = static_cast<const Jet*>(in);
    Jet*       b = static_cast<Jet*>(inout);
    for (int i = 0; i < *len; ++i) b[i] += a[i];   // otinum::operator+=
}

MPI_Op MPI_OTI_SUM;
MPI_Op_create(&jet_sum, /*commute=*/1, &MPI_OTI_SUM);
MPI_Allreduce(&local, &global, 1, MPI_OTINUM, MPI_OTI_SUM, MPI_COMM_WORLD);
```

The operator is just `otinum::operator+=` — OTI arithmetic plugs straight into
MPI. For a **sum**, this happens to equal `MPI_SUM` over the raw coefficients
(jet addition is coefficient-wise), so you could reduce the underlying doubles
with `MPI_SUM` instead. The custom op is the *general* mechanism: it's required
the moment the combine is not coefficient-wise (e.g. reducing a **product** of
jets, where `operator*` is a convolution), and it keeps the reduction in
`MPI_OTINUM` units, consistent with the rest of the section.

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
