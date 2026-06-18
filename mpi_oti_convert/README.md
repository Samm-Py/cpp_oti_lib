# mpi_oti_convert — OTI-enabling a Kokkos + MPI program

A before/after pair showing exactly what changes to add OTI derivatives to an
existing `double` Kokkos + MPI code. Both evaluate `model(x,y) = sin(x)·exp(y)`
over a distributed 1000×1000 grid (one block per rank, on the Kokkos device) and
gather the field to rank 0.

- **`convert_before.cpp`** — plain `double`. Prints a sample value.
- **`convert_after.cpp`** — the same program with `otinum`. Prints the same value
  plus `d/dx` and `d/dy`.

The five source changes (and the build flags) are walked through in the
*Converting a Kokkos + MPI Program to OTI* tutorial. The headline: the `model()`
kernel and the MPI/Kokkos plumbing are unchanged — only the scalar type, the
input seeding, the MPI datatype, and the result readout differ.

## Build & run

```sh
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/root/Research/kokkos-cuda-install/bin/nvcc_wrapper \
  -DKokkos_ROOT=/root/Research/kokkos-cuda-install
cmake --build build
mpirun -np 4 ./build/convert_before
mpirun -np 4 ./build/convert_after
```

`convert_before` uses no OTI headers; `convert_after` defines `OTI_ENABLE_KOKKOS`
and adds the library include path (see its `CMakeLists.txt` target). Verified the
value matches and the derivatives appear, np=1/2/4 on a GTX 1650 with Intel MPI.
