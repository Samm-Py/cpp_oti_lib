Across GPUs
===========

The GPU mirror of :doc:`cpu`: the standard way to run an MPI + Kokkos code across
GPUs is **one rank per GPU**:
each rank binds to its own device, computes its slice of the problem there, and
the slices are combined over MPI. This tutorial builds that pattern and runs it on
a single-GPU machine by *simulating* the multi-GPU case -- each rank takes an
exclusive turn on the one physical device.

The source is ``mpi_oti_gpu_toy/mpi_oti_multigpu.cpp``. It assumes you can already
build the CUDA Kokkos backend; see :doc:`../kokkos_gpu` for the setup.

Binding Each Rank To A Device
-----------------------------

A rank chooses its device as ``rank % (GPUs on the node)`` and passes that id to
Kokkos at initialization. This is the real binding code -- on a four-GPU node
ranks 0–3 land on distinct devices; on a one-GPU node every rank maps to device
0:

.. code-block:: cpp

   int num_gpus = 0;
   cudaGetDeviceCount(&num_gpus);
   const int my_device = num_gpus > 0 ? rank % num_gpus : 0;

   Kokkos::InitializationSettings settings;
   settings.set_device_id(my_device);
   Kokkos::initialize(settings);

Simulating Multiple GPUs: The Token Ring
----------------------------------------

On real multi-GPU hardware the ranks now just run -- distinct devices proceed
concurrently. With only one physical GPU, several ranks would otherwise contend
for it at once. To make each rank behave as if it owned its device, we serialize
access with a **token ring**: a rank blocks until it receives the token from the
previous rank, does its device work while it alone holds the token, fences, then
passes the token on. "Wait until the device is free, use it, release it."

.. code-block:: cpp

   int token = 0;
   if (rank > 0)
       MPI_Recv(&token, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD, &status);

   // --- this rank now holds the GPU to itself ---
   Kokkos::View<Jet*> d_local("d_local", count);
   Kokkos::parallel_for("evaluate", count,
       KOKKOS_LAMBDA(int k) { d_local(k) = evaluate(start + k); });
   Kokkos::fence();        // device work must finish before the GPU is "free"

   // --- release: hand the GPU to the next rank ---
   if (rank < size - 1)
       MPI_Send(&token, 1, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD);

The ``Kokkos::fence()`` is essential: device kernels are asynchronous, so without
it a rank would pass the token while its kernel is still running, breaking the
exclusivity the ring is supposed to guarantee.

Build And Run
-------------

Configure with the Kokkos ``nvcc_wrapper`` and point at the CUDA Kokkos install
(same as :doc:`../kokkos_gpu`):

.. code-block:: console

   cd mpi_oti_gpu_toy
   cmake -S . -B build \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
     -DKokkos_ROOT=/path/to/kokkos-cuda-install
   cmake --build build --target mpi_oti_multigpu
   mpirun -np 4 ./build/mpi_oti_multigpu

Each rank prints when it takes its turn, so the output appears in ring order:

.. code-block:: text

   rank 0/4: bound to GPU 0 of 1 | exclusive device turn:   3.30 ms for 250000 jets
   rank 1/4: bound to GPU 0 of 1 | exclusive device turn:   2.93 ms for 250000 jets
   rank 2/4: bound to GPU 0 of 1 | exclusive device turn:   2.95 ms for 250000 jets
   rank 3/4: bound to GPU 0 of 1 | exclusive device turn:   3.25 ms for 250000 jets

   4 ranks took exclusive turns on 1 physical GPU(s); assembled 1000000 jets.
   verify: PASS (bit-exact) (0 mismatches)
   note: 4 ranks shared 1 GPU via the token ring (serialized). On a 4-GPU node
   they would run concurrently.

After the ring, the per-rank results are combined with one ``MPI_Gatherv`` using
the same ``oti::mpi::make_datatype<Jet>()`` from :doc:`make_datatype` (the
gather is host-staged here; the device-pointer alternative for CUDA-aware MPI is
covered in :doc:`integration`). Rank 0 verifies the assembled grid bit-for-bit.

What This Does And Does Not Show
--------------------------------

The MPI + Kokkos *structure* -- bind rank to device, compute your slice there,
combine over MPI -- is exactly what you would ship for a real multi-GPU run. Only
the token ring is the single-GPU stand-in.

It does **not** show a speedup: serializing turns on one GPU is, if anything,
slower than letting the ranks share it, because the point is exclusivity, not
throughput. The payoff is real on a multi-GPU node, where you **remove the token
ring** and the distinct devices run concurrently -- the device binding and the
MPI gather stay identical.

.. note::

   When several ranks share one device without a token ring they time-slice it,
   which is correct but contended. The token ring is the simplest way to give
   each rank a clean, exclusive turn so the single-GPU run faithfully mirrors the
   one-rank-per-GPU execution model.
