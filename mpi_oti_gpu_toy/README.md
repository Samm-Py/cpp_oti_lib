# mpi_oti_gpu_toy — moving GPU-produced OTI jets over MPI

The GPU counterpart of `../mpi_oti_toy`. It evaluates `f(x,y)=sin(x)·exp(y)`
over a 1000×1000 grid of `otinum<2,2,double>` jets **on the device** (Kokkos +
CUDA), then gathers every jet to rank 0 with the same
`oti::mpi::make_datatype<Jet>()` datatype — proving the datatype works for
device-originated data.

## What it demonstrates

- **The datatype is GPU-agnostic.** `otinum`'s layout (`sizeof`, alignment,
  member order) is identical in host and device memory — the toy checks this
  directly (a device kernel reports `sizeof(Jet)`, compared to the host value).
  So `MPI_Type_contiguous(ncoeffs, MPI_DOUBLE)` describes a device buffer just as
  well as a host one; nothing in `make_datatype` changes for GPU.
- **Host-staging path.** On this box MPI (Intel MPI) is not CUDA-aware for the
  NVIDIA device, so each rank `deep_copy`s its device results to a host mirror
  and MPI gathers the host buffer. A CUDA-aware MPI build would instead pass the
  device pointer straight to `MPI_Gatherv` — same datatype, no other change.
- **Verification is device-vs-device.** GPU `sin`/`exp` need not match the CPU
  bit-for-bit, so rank 0 recomputes the whole grid **on its GPU** and `memcmp`s
  against the gathered buffer. Both run the same kernel on the same hardware, so
  any mismatch is an MPI-transport bug — which is what this toy tests.

## Build & run

Needs a CUDA-enabled Kokkos install and MPI. Configure with Kokkos's
`nvcc_wrapper` as the C++ compiler:

```sh
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/root/Research/kokkos-cuda-install/bin/nvcc_wrapper \
  -DKokkos_ROOT=/root/Research/kokkos-cuda-install
cmake --build build
mpirun -np 2 ./build/mpi_oti_gpu_toy     # also valid at np=1 and np>2
```

Rank 0 prints the backend, the host/device `sizeof` check, a sample jet, and the
bit-exact verdict. Verified `PASS` at np=1/2/4 on a GTX 1650 (sm_75) with Intel
MPI; all ranks share the single GPU, which is why the device-vs-device check is
bit-exact.
