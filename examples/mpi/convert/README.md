# mpi_oti_convert — OTI-enabling an MPI program (movement collectives round trip)

A before/after pair showing exactly what changes to add OTI derivatives to an
existing `double` MPI code. Both do the realistic data-movement round trip over a
distributed 1-D field of `N = 1000` points:

```
root builds input field ──Bcast(p)──▶ all ranks
                        ──Scatterv──▶ in_local        (root → ranks)
        each rank: out_local = sin(p · in_local)      (compute, no comm)
                        ──Gatherv───▶ out on root      (ranks → root)
```

- **`convert_before.cpp`** — plain `double`. Prints a sample value.
- **`convert_after.cpp`** — the same program with `otinum`. Prints the same value
  plus `d/du` (field sensitivity, seeded per point) and `d/dp` (parameter
  sensitivity, seeded and delivered via `MPI_Bcast`).

Plain MPI — **no Kokkos, no GPU** — to keep the first rung as simple as possible.
For an application already configured for Kokkos, the additional OTI-specific
setting is `OTI_ENABLE_KOKKOS`; backend and compiler setup are covered in the
*Integration* reference.

The five source changes are walked through in the *Independent Evaluation*
tutorial. The headline: the `transform()` kernel and the MPI plumbing are
byte-for-byte unchanged — only the scalar type, the input seeding, the MPI
datatype, and the result readout differ.

## Movement needs only the datatype

`Bcast`, `Scatterv`, and `Gatherv` only *move* jets; they never combine them. So
one committed datatype (`oti::mpi::make_datatype<Scalar>()`) serves all three —
**no custom `MPI_Op`**. Here the per-element computation is unchanged, so moving
the finished jets introduces no rounding and the gathered field is
**bit-identical** to the serial run at the tested rank counts. (Combining
collectives — `Reduce`/`Allreduce` — are the other family and need the operator
from the `mpi_oti_reduce` example.)

## Build & run

```sh
cmake -S . -B build && cmake --build build
mpirun -np 4 ./build/convert_before
mpirun -np 4 ./build/convert_after
```

Or directly:

```sh
mpicxx -std=c++17 -O2 convert_before.cpp -o convert_before
mpicxx -std=c++17 -O2 -I ../../../include convert_after.cpp -o convert_after
```

Verified the value matches and the derivatives appear, identical at np=1/4
(movement collectives are rank-count-invariant).
