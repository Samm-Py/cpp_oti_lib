# MPI + OTI with device-resident buffers

The fifth conversion rung: not a new communication pattern, but the same
patterns (movement *and* combining) when the jet buffers live in GPU memory.
The point of the example is what does **not** change: the committed datatype
from `oti::mpi::make_datatype`, the custom reduction op from
`oti::mpi::make_sum_op`, and every count/displacement are identical whether
the pointers handed to MPI are host or device memory.

Each rank evaluates its block of `f(x, y; a, b) = sin(a·x)·exp(b·y)` on the
device with the design parameters `(a, b)` seeded as OTI variables, reduces the
block to one partial-sum jet **on the device** (`Kokkos::parallel_reduce` over
jets, via a two-line `Kokkos::reduction_identity` specialization), and then
runs `MPI_Gatherv` + `MPI_Allreduce` over two transports:

- **host staging** — `deep_copy` to host mirrors, MPI on host buffers. Works
  under any MPI; this is the portable baseline (and the only path taken when
  the MPI is not CUDA-aware).
- **device direct** — the device pointers go straight into MPI (CUDA-aware
  MPI, detected at runtime via `MPIX_Query_cuda_support`; override with
  `OTI_MPI_DEVICE=1/0`).

The two transports are checked against each other **bit-for-bit**; the staged
QoI is additionally verified against a serial host recompute to a tight
relative tolerance, and the raw-coefficient `MPI_SUM` escape hatch for jet
sums is checked against the custom op. The program returns nonzero if any
check fails.

One behavioural fact this example pins down: an `MPI_Op` is a **host**
callback by the MPI standard, on every implementation. A CUDA-aware MPI
therefore stages device buffers through host internally before invoking the
jet-sum callback — correct (verified bit-exact here), but not device-native.
Movement collectives, by contrast, move device bytes natively. For a sum, the
device-native alternative is the escape hatch: `MPI_SUM` over
`count * ncoeffs` raw doubles, no callback at all.

Tutorial: `docs/tutorials/mpi/converting/device.rst`. Build and run: see
`CMakeLists.txt`.
