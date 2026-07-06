Device Buffers (CUDA-Aware MPI)
===============================

The last rung is different in kind. Every previous rung changed *what* the ranks
communicate -- movement, combining, structured halos, unstructured ghost lists.
This one changes *where the buffers live*: GPU memory. The point of the example
is what does **not** change -- which is everything on the OTI side. The committed
datatype describes a jet in device memory exactly as it does in host memory (the
layout is identical), the custom reduction op folds them the same way, and every
count and displacement keeps its value. Converting a device-resident MPI code to
OTI is the same five edits as :doc:`index`; whether the pointers handed to MPI
are host or device is an independent, orthogonal choice.

The source is ``examples/mpi/device/main.cpp``. It runs the two collectives the
earlier rungs established -- ``MPI_Gatherv`` (movement) and ``MPI_Allreduce``
with the custom jet-sum op (combining) -- over the same device-computed data
twice, once per transport, and requires the results to agree bit for bit.

The Starting Point: Host Staging
--------------------------------

A Kokkos code whose jets live in a device ``View`` can always talk to MPI the
portable way: ``deep_copy`` to a host mirror, then the familiar host-buffer
collectives. This works under **any** MPI, including ones that are not
CUDA-aware, and it is the pattern the :doc:`../../integration` tutorial builds
on:

.. code-block:: cpp

   Kokkos::View<Jet*> d_local("d_local", count);          // computed on the GPU
   // ... parallel_for fills d_local ...

   auto h_local = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_local);
   MPI_Gatherv(h_local.data(), count, MPI_OTINUM, ...);   // MPI sees host memory

The Change
----------

Under a CUDA-aware MPI the staging lines simply disappear -- the device pointer
goes straight into the same call:

.. code-block:: diff

   -auto h_local = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_local);
   -MPI_Gatherv(h_local.data(), count, MPI_OTINUM, ...);
   +MPI_Gatherv(d_local.data(), count, MPI_OTINUM, ...);  // device pointer straight in

Note what is absent from the diff: the datatype, the op, the counts, and the
displacements. Nothing OTI-specific knows or cares where the buffer lives.

Movement Vs Combining On The Device
-----------------------------------

The two collective families from :doc:`index` behave differently once buffers
are device-resident, and the difference is worth knowing:

**Movement collectives are device-native.** ``Scatterv`` / ``Gatherv`` /
``Bcast`` / ``Sendrecv`` only move bytes, so a CUDA-aware MPI transfers device
buffers directly (GPUDirect / CUDA IPC where available). The committed jet
datatype -- including the derived ``MPI_Type_vector`` / ``MPI_Type_indexed``
halos built on it -- describes device memory unchanged.

**Predefined reductions can be device-native too.** On modern stacks
(UCC/HCOLL-backed Open MPI, MVAPICH2-GDR, NCCL-backed collectives) a
``MPI_SUM`` over ``MPI_DOUBLE`` executes its arithmetic *in GPU kernels* and
moves data GPU-to-GPU over GPUDirect RDMA / NVLink -- the buffers never touch
host memory at all.

**Custom-op reductions run their operator on the host.** An ``MPI_Op`` created
with ``MPI_Op_create`` is a *host* function pointer with a type-erased ABI --
the implementation receives opaque compiled host code that it cannot move to
the GPU, and the accelerated backends (NCCL, UCC) only cover their fixed set
of predefined operations and datatypes. So a user op always drops the
reduction off the device-native path: a CUDA-aware MPI stages the device
buffers through host internally before invoking the jet-sum callback, and the
result is **correct** -- same jets, same reduction tree, so the example
requires it to be bit-identical to the staged transport -- but the reduction
itself is not device-native. This is a limit of today's ``MPI_Op`` interface,
not of the mathematics: a combine that decomposes into predefined ops can ride
the device-native path (the sum escape hatch below), an extremum can be done
in two device-native phases (predefined ``MPI_MAXLOC`` on ``(real value,
rank)`` pairs, then ``Bcast`` the winning jet from that rank), and in the
limit any associative combine can stay fully device-resident by hand-rolling
the reduction tree -- pairwise device-buffer ``Sendrecv`` rounds with an
on-device Kokkos kernel applying the combine between rounds. Two practical
consequences:

* Reduce large device data *locally on the device first* (the example uses a
  ``Kokkos::parallel_reduce`` over jets), so what crosses the reduction is one
  partial-sum jet per rank, not a field. This is good practice on host-only
  codes too; on device buffers it is what keeps the hidden staging negligible.
* For a **sum** specifically there is a fully device-native escape hatch: jet
  addition is coefficient-wise, so the predefined ``MPI_SUM`` over the raw
  coefficient block is the same mathematics with no user callback at all --

  .. code-block:: cpp

     // equivalent to the custom op, for SUM only -- no host callback:
     MPI_Allreduce(d_partial.data(), d_qoi.data(),
                   n * Jet::ncoeffs, MPI_DOUBLE, MPI_SUM, comm);

  This does *not* generalize: a jet **product** is a convolution across
  coefficients and ``maxloc`` / ``minloc`` are not coefficient-wise either, so
  those genuinely need the custom op (and, on device buffers, its internal
  staging). They are inherently small-count QoI reductions, so that staging is
  cheap. MPI may pick a different reduction algorithm for a predefined op, so
  compare the escape hatch to the custom op with a tolerance, not ``memcmp``.

If a custom-op reduction on device pointers ever meets an MPI stack whose
CUDA support is broken or absent, the failure is an immediate fault at the
reduction call -- not silent corruption. The callback in ``otinum/mpi.hpp``
copies each element through an aligned local before doing jet arithmetic, so a
device pointer that reaches it faults on the first byte.

Reducing A Jet Field On The Device
----------------------------------

The one genuinely new line in the example is not MPI at all. To
``parallel_reduce`` over a custom scalar, Kokkos needs its additive identity:

.. code-block:: cpp

   namespace Kokkos {
   template <>
   struct reduction_identity<Jet> {
       KOKKOS_FORCEINLINE_FUNCTION static Jet sum() { return Jet(0.0); }
   };
   }

   Kokkos::View<Jet> d_partial("d_partial");      // stays in device memory
   Kokkos::parallel_reduce(
       "partial_sum", count,
       KOKKOS_LAMBDA(long k, Jet& acc) { acc += d_local(k); }, d_partial);

The ``Sum`` reducer starts each thread at that identity and joins partials with
``operator+=``, which the jet already provides. The result view is
device-resident, so the subsequent ``Allreduce`` really does receive a device
pointer.

Runtime Detection And Fallback
------------------------------

CUDA-awareness is a property of the MPI *build*, so the example picks its
transport at runtime and keeps the staged path as the fallback -- the pattern a
portable application should ship:

.. code-block:: cpp

   static bool detect_cuda_aware()
   {
       bool aware = false;
   #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
       aware = (MPIX_Query_cuda_support() == 1);   // Open MPI extension
   #endif
       if (const char* e = std::getenv("OTI_MPI_DEVICE")) {
           aware = (std::atoi(e) != 0);            // manual override
       }
       return aware;
   }

Under a host-parallel Kokkos backend (OpenMP / Serial) the "device" views are
host memory anyway, so ``OTI_MPI_DEVICE=1`` safely exercises the direct path
under any MPI.

Build And Run
-------------

The example needs Kokkos and MPI; for a CUDA backend, configure with the Kokkos
``nvcc_wrapper`` and (to actually take the device path) a CUDA-aware MPI such
as an Open MPI built ``--with-cuda``:

.. code-block:: console

   cd examples/mpi/device
   cmake -S . -B build \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
     -DKokkos_ROOT=/path/to/kokkos \
     -DMPI_HOME=/path/to/cuda-aware-openmpi
   cmake --build build
   mpirun -np 4 ./build/mpi_oti_device

.. code-block:: text

   backend            : Cuda
   ranks              : 4
   transport          : device pointers (CUDA-aware MPI) + staged reference
   grid               : 1000 x 1000  (1000000 jets, 45.8 MB)
   QoI = mean of sin(a x) exp(b y)  at a=1.000, b=1.000
     value            =  0.7898879935
     d/da             =  0.6558559384
     d/db             =  0.4598239459
   staged vs serial   : PASS (max relative diff 3.08e-14)
   device Gatherv     : PASS (bit-exact vs staged)
   device custom-op   : PASS (bit-exact vs staged)
   raw MPI_SUM hatch  : PASS (max relative diff 0.00e+00)

The QoI and both first-order sensitivities match the :doc:`reduce` rung's output
exactly -- it is the same problem, computed on the GPU instead of the host. The
program returns nonzero if any check fails, so it is CI-gateable; under a
non-CUDA-aware MPI it reports the device transport as skipped and verifies the
staged path only.

.. note::

   **Functionality and performance are separate claims.** These checks verify
   that device-buffer transport is *correct*. Whether it is *faster* than
   staging depends on the fabric: GPUDirect RDMA across nodes, CUDA IPC within
   a node. On a platform without those (this page's reference run is WSL2,
   which lacks CUDA IPC), a CUDA-aware MPI moves device buffers by staging them
   internally -- same result, no code change, but no transport speedup to
   measure. Benchmark on the target cluster before attributing performance to
   either path.
