# mpi_oti_toy — moving OTI numbers over MPI

The minimal possible MPI + `cpp_oti_lib` program. It evaluates a function over a
1000×1000 grid, **embarrassingly parallel** (no ghost nodes, no halo exchange),
and gathers every jet back to rank 0 using a *committed* MPI datatype for one
`otinum`.

## What it demonstrates

The only new idea is that **MPI can move an `otinum<M,N>` as a first-class
element**, because the jet is a fixed-size contiguous block of `ncoeffs`
coefficients with no pointers/heap:

```cpp
#include "otinum/mpi.hpp"
MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();  // contiguous(ncoeffs, MPI_DOUBLE), committed
// ... use it ...
oti::mpi::free_datatype(MPI_OTINUM);
```

`oti::mpi::make_datatype<T>()` (in the optional `otinum/mpi.hpp`, which is *not*
part of the umbrella so non-MPI builds carry no dependency) builds and commits
`MPI_Type_contiguous(Jet::ncoeffs, MPI_DOUBLE)` and `static_assert`s that the jet
is tightly packed — so consumers don't re-roll those two lines or the
no-padding check by hand.

Once committed, gather counts are expressed in **natural units** ("N jets"),
not bytes — and the same `MPI_OTINUM` is the building block for the derived
datatypes a real solver needs later (`MPI_Type_vector` for strided column halos,
`MPI_Type_indexed` for unstructured ghost-node lists).

Each grid point seeds its coordinates as infinitesimals
(`Jet::variable(0, x0)`, `Jet::variable(1, y0)`) and evaluates
`f(x,y) = sin(x)·exp(y)`, so every gathered jet carries `f` plus its full
gradient and Hessian (as normalized Taylor coefficients).

## Correctness dependency

`MPI_Type_contiguous(ncoeffs, MPI_DOUBLE)` only matches the array stride if the
jet has no trailing alignment padding. `make_datatype<T>()` carries that check
itself — a `static_assert(oti::mpi::is_tightly_packed_v<T>)` — so every consumer
gets it for free. (otinum is tightly packed for every shape by construction; if
that ever changed, the fix is `MPI_Type_create_resized`, not removing the assert.)

## Build & run

```sh
mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_toy
mpirun -np 4 ./mpi_oti_toy
# or via CMake:
cmake -S . -B build && cmake --build build && mpirun -np 4 ./build/mpi_oti_toy
```

Rank 0 prints a sample jet and verifies the gathered grid **bit-for-bit**
against a single-process recompute. Verified with Intel MPI on `np = 1, 4, 7`
(7 exercises the uneven `MPI_Gatherv` path).

## Datatype confidence test

`test_mpi_datatype.cpp` is a focused check that `make_datatype` matches the C++
otinum layout and transports jets faithfully, across the cases the toy doesn't
cover — **float as well as double, and odd-`ncoeffs` shapes** (`<4,1>`: only 4/8-
aligned, where a padding surprise would bite). For each shape it asserts
`MPI_Type_size == ncoeffs*sizeof(Coeff)`, the datatype **extent == `sizeof(T)`**
(what makes `count>1` / `Gatherv` stride correctly), and a ring `Sendrecv` of 257
jets round-trips bit-exact. Returns nonzero on any failure, so it works as a CI
gate.

```sh
mpicxx -std=c++17 -O2 -I ../include test_mpi_datatype.cpp -o test_mpi_datatype
mpirun -np 2 ./test_mpi_datatype     # also valid at np=1 and np>2
```

All shapes PASS at np=1/2/4 (Intel MPI).
