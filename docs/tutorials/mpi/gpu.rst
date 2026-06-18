MPI With GPU-Resident Jets
==========================

This tutorial runs the same gather as :doc:`make_datatype`, but the jets are
produced on a CUDA device. The point is to show two things: the committed
datatype does not change at all for GPU data, and the data plumbing (how the
buffers reach MPI) is the only thing that does.

The source is ``mpi_oti_gpu_toy/`` at the repository root. It assumes you can
already build the CUDA Kokkos backend; see :doc:`../kokkos_gpu` for the toolkit,
host-compiler, and architecture-flag setup.

The Datatype Is GPU-Agnostic
----------------------------

An ``otinum``'s object representation -- ``sizeof``, alignment, and member order
-- is identical in host and device memory, because the alignment rule is a
``constexpr`` evaluated the same way by NVCC and the host compiler. So
``MPI_Type_contiguous(ncoeffs, MPI_DOUBLE)`` describes a device buffer exactly as
well as a host one, and ``oti::mpi::make_datatype<Jet>()`` is used unchanged. The
toy checks the layout agreement directly -- a device kernel reports
``sizeof(Jet)``, compared against the host value:

.. code-block:: text

   sizeof(Jet) host/device : 48 / 48  OK

Host Staging vs CUDA-Aware MPI
------------------------------

What changes for GPU is *where the buffer lives* when it reaches MPI. There are
two paths:

* **CUDA-aware MPI** passes the device pointer straight to the MPI call; the MPI
  implementation moves device memory itself (potentially over GPUDirect). It
  needs an MPI built with CUDA support.
* **Host staging** copies the device buffer to a host mirror, calls MPI on the
  host buffer, and copies back on receive. It always works, at the cost of the
  extra device-host transfers.

This toy uses host staging, because the MPI on the development box is not
CUDA-aware for the NVIDIA device. Each rank evaluates its slice on the device,
then ``deep_copy``\ s to a host mirror before the gather:

.. code-block:: cpp

   Kokkos::View<Jet*> d_local("d_local", count);
   Kokkos::parallel_for("evaluate", count,
       KOKKOS_LAMBDA(int k) { d_local(k) = evaluate(start + k); });

   auto h_local = Kokkos::create_mirror_view(d_local);
   Kokkos::deep_copy(h_local, d_local);          // device -> host

   MPI_Gatherv(h_local.data(), count, MPI_OTINUM, /* ... */);

A CUDA-aware build would drop the mirror and pass ``d_local.data()`` directly --
the datatype and the gather call are otherwise unchanged.

Verifying Without Tripping On GPU Math
--------------------------------------

GPU ``sin``/``exp`` need not match the CPU bit-for-bit, so comparing
device-produced jets against a *host* recompute would test the math libraries,
not the transport. Instead rank 0 recomputes the whole grid **on its GPU** and
compares against the gathered buffer. Both sides run the same kernel on the same
hardware, so they are bit-identical, and any mismatch is therefore a transport
bug -- which is what this toy actually tests.

Build And Run
-------------

The GPU toy is a standalone CMake project. Configure it with the Kokkos
``nvcc_wrapper`` as the C++ compiler and point it at the CUDA Kokkos install:

.. code-block:: console

   cd mpi_oti_gpu_toy
   cmake -S . -B build \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
     -DKokkos_ROOT=/path/to/kokkos-cuda-install
   cmake --build build
   mpirun -np 2 ./build/mpi_oti_gpu_toy

Rank 0 prints the backend, the layout check, a sample jet computed on the device,
and the bit-exact verdict:

.. code-block:: text

   backend          : Cuda
   ranks            : 2 (host-staged gather)
   grid             : 1000 x 1000 (1000000 points)
   sizeof(Jet) host/device : 48 / 48  OK
   sample @ 500500     : value= 0.79155923  d/dx= 1.44721739  d/dy= 0.79155923
   verify (device recompute) : PASS (bit-exact) (0 mismatching jets)

Verified ``PASS`` at ``np = 1, 2, 4`` on a GTX 1650 (Turing, ``sm_75``). All
ranks share the single GPU, which is why the device-vs-device check is bit-exact.

.. note::

   When several ranks share one physical GPU they time-share it, which is fine
   for correctness but not for performance. On a real multi-GPU cluster each rank
   binds to its own device; the datatype and the staging logic are unchanged.
