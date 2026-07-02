# mpi_oti_gpu_toy — OTI jets on the GPU under MPI

The GPU counterpart of `../toy`. Two programs, both evaluating
`f(x,y)=sin(x)·exp(y)` over a 1000×1000 grid of `otinum<2,2,double>` jets **on the
device** (Kokkos + CUDA) and combining the results over MPI with the same
`oti::mpi::make_datatype<Jet>()` datatype.

## `mpi_oti_multigpu` — the one-rank-per-GPU pattern (tutorial: *Multi-GPU Execution*)

The standard multi-GPU layout: each rank binds to its own device
(`rank % num_gpus`, via `Kokkos::InitializationSettings::set_device_id`), computes
its slice there, and the slices are gathered over MPI. On this single-GPU box it
*simulates* the multi-GPU case with a **token ring** — a rank blocks until it
receives the token, does its device work exclusively, `fence`s, then passes the
token on ("wait until the device is free, use it, release it"). The MPI + Kokkos
structure is exactly what a real multi-GPU run ships; on true multi-GPU hardware
you drop the token ring and the distinct devices run concurrently.

```sh
cmake --build build --target mpi_oti_multigpu
mpirun -np 4 ./build/mpi_oti_multigpu
```

## `mpi_oti_gpu_toy` — the transport reference (device-pointer vs host staging)

Demonstrates how device-resident jets reach MPI:

- **The datatype is GPU-agnostic.** `otinum`'s layout (`sizeof`, alignment, member
  order) is identical in host and device memory — the program checks this directly
  (a device kernel reports `sizeof(Jet)`, compared to the host value). So
  `MPI_Type_contiguous(ncoeffs, MPI_DOUBLE)` describes a device buffer unchanged.
- **Dual-path transport, chosen at runtime.** It queries
  `MPIX_Query_cuda_support()` (override with `OTI_MPI_DEVICE=1/0`): if the MPI is
  CUDA-aware it gathers straight from the device pointer, otherwise it stages
  through a host mirror. Same datatype and gather call either way.
- **Verification is device-vs-device.** GPU `sin`/`exp` need not match the CPU
  bit-for-bit, so rank 0 recomputes the grid **on its GPU** and `memcmp`s against
  the gathered buffer — any mismatch is then a transport bug, which is what this
  tests.

```sh
mpirun -np 2 ./build/mpi_oti_gpu_toy     # also valid at np=1 and np>2
```

## Build

Both targets need a CUDA-enabled Kokkos install and MPI. Configure with Kokkos's
`nvcc_wrapper` as the C++ compiler:

```sh
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/root/Research/kokkos-cuda-install/bin/nvcc_wrapper \
  -DKokkos_ROOT=/root/Research/kokkos-cuda-install
cmake --build build
```

Both verified `PASS` (bit-exact) at np=1/2/4 on a GTX 1650 (sm_75) with Intel MPI;
all ranks share the single GPU. On this box the transport program detects the MPI
as not CUDA-aware (WSL2 limitation — see the *Integration* tutorial) and uses host
staging.
