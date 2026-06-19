Integrating cpp_oti_lib Into An MPI + Kokkos Application
========================================================

This guide is the culmination of the MPI section: how to bring ``cpp_oti_lib``
into your own simulation code when you are already (or want to be) running on
**MPI** for distribution and **Kokkos** for on-node / GPU parallelism. It pulls
together the datatype helper from :doc:`make_datatype` and the device transport
pattern (`Step 4 -- The MPI + GPU Transport Pattern`_), and ends with the
toolchain pitfalls that actually bite when you stack these three pieces.

The Dependency Model
--------------------

The single most important thing to understand is that **cpp_oti_lib depends on
neither MPI nor Kokkos**. It is header-only, and it *interoperates* with both
rather than requiring them:

.. code-block:: text

   your application code
        |        |              <- your real dependencies
     Kokkos     MPI
        |        |
   cpp_oti_lib (header-only)    <- adapts to whatever you already link
     - core headers:     plain C++17, no dependencies
     - OTI_ENABLE_KOKKOS: makes otinum device-callable (uses Kokkos::Array,
                          annotates arithmetic for device kernels)
     - otinum/mpi.hpp:    one committed MPI_Datatype per jet shape

So you do not "add cpp_oti_lib and then get MPI and Kokkos." You bring your own
Kokkos and MPI; ``cpp_oti_lib`` is a thin header layer that knows how to ride on
them. The scalar library works with neither installed; each optional piece turns
on only when you opt in.

Step 1 -- Add OTI To Your Kernels
---------------------------------

The core move is to replace the scalar type your kernel computes in with an
``otinum``. A kernel written against a ``Scalar`` template parameter (or a
typedef) needs no logic changes -- the overloaded arithmetic and ``<cmath>``
surface carry the derivatives through. See :doc:`../basic_usage` and
:doc:`../directional_derivatives` for the seeding patterns. Nothing here involves
MPI or Kokkos yet; this step works in ordinary serial C++.

Step 2 -- Make It Device-Callable (Kokkos)
------------------------------------------

Define ``OTI_ENABLE_KOKKOS``. This switches the coefficient container from
``std::array`` to ``Kokkos::Array`` and annotates the arithmetic so it is
callable inside device kernels. An array of jets is then a ``Kokkos::View<Jet*>``
and your ``parallel_for`` body computes jets on the device exactly as it would on
the host (see :doc:`../kokkos_gpu`). The CUDA build requires Kokkos's
``nvcc_wrapper`` as the C++ compiler.

Step 3 -- Distribute It (MPI)
-----------------------------

Include the optional ``otinum/mpi.hpp`` and commit one datatype per jet shape:

.. code-block:: cpp

   #include "otinum/mpi.hpp"

   MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
   // ... sends / receives / collectives in units of MPI_OTINUM ...
   oti::mpi::free_datatype(MPI_OTINUM);

Because a jet is a contiguous, padding-free block of coefficients, this datatype
is all MPI needs -- there is no serialization layer. The same ``MPI_OTINUM`` is
the base element for the derived datatypes a real solver needs (``MPI_Type_vector``
for strided halos, ``MPI_Type_indexed`` for unstructured ghost-node lists). See
:doc:`make_datatype` for the full treatment and the layout confidence test.

Step 4 -- The MPI + GPU Transport Pattern
-----------------------------------------

When the jets live on a device *and* you distribute them, the question is how the
buffer reaches MPI. First, the good news: the committed datatype does **not**
change for GPU data. An ``otinum``'s layout (``sizeof``, alignment, member order)
is identical in host and device memory, so ``MPI_Type_contiguous(ncoeffs,
MPI_DOUBLE)`` describes a device buffer exactly as well as a host one. There are
two ways to hand that buffer to MPI:

* **CUDA-aware MPI** passes the device pointer straight to the MPI call; the MPI
  implementation moves device memory itself (potentially over GPUDirect). It
  needs an MPI built *and* initialized with CUDA support.
* **Host staging** copies the device buffer to a host mirror, calls MPI on the
  host buffer, and copies back on receive. It always works, at the cost of the
  extra device-host transfers.

**Detect which is available at runtime and fall back**, rather than assuming:

.. code-block:: cpp

   bool cuda_aware = false;
   #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
       cuda_aware = (MPIX_Query_cuda_support() == 1);   // Open MPI extension
   #endif

   if (cuda_aware) {
       MPI_Gatherv(d_local.data(), count, MPI_OTINUM, /* device recv ... */);
   } else {
       auto h_local = Kokkos::create_mirror_view(d_local);
       Kokkos::deep_copy(h_local, d_local);             // stage to host
       MPI_Gatherv(h_local.data(), count, MPI_OTINUM, /* host recv ... */);
   }

The datatype and the MPI call are identical on both branches -- only the buffer
differs. ``mpi_oti_gpu_toy/main_gpu.cpp`` is a complete worked example that runs
either way. The reason to use a *runtime* check and not a build-time assumption is
in `Toolchain Gotchas`_ below.

Building It -- The CMake Recipe
-------------------------------

A consumer ``CMakeLists.txt`` for the full stack. ``find_package(otinum)``
provides the header-only interface target ``oti::otinum_headers``; if that
install was configured with ``OTI_ENABLE_KOKKOS=ON``, the target already carries
the ``OTI_ENABLE_KOKKOS`` define and links ``Kokkos::kokkos`` for you.

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(my_oti_app CXX)

   find_package(otinum REQUIRED)   # header-only; target oti::otinum_headers
   find_package(Kokkos REQUIRED)   # your on-node / GPU backend
   find_package(MPI REQUIRED)      # your distribution layer

   add_executable(my_app main.cpp)
   target_link_libraries(my_app PRIVATE
       oti::otinum_headers
       Kokkos::kokkos
       MPI::MPI_CXX)

Configure it with the Kokkos compiler wrapper so the device path compiles, and
point at your Kokkos and (if not on the default search path) MPI installs:

.. code-block:: console

   cmake -S . -B build \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
     -DKokkos_ROOT=/path/to/kokkos-cuda-install \
     -DMPI_CXX_COMPILER=/path/to/your-mpi/bin/mpicxx
   cmake --build build

See :doc:`../cmake_package` for ``find_package(otinum)`` on its own, and
:doc:`../kokkos_gpu` for the CUDA Kokkos install and architecture flags.

Toolchain Gotchas
-----------------

Stacking three build systems is where integration actually goes wrong. These are
the failures worth knowing in advance.

**nvcc_wrapper plus an MPI compiler wrapper.** The CUDA build needs
``nvcc_wrapper`` as ``CMAKE_CXX_COMPILER`` (for Kokkos device code), but MPI flags
normally come from an MPI compiler wrapper. Let CMake's ``FindMPI`` extract the
MPI include/link flags (point it at your ``mpicxx`` via ``MPI_CXX_COMPILER``) and
apply them to ``nvcc_wrapper`` through the ``MPI::MPI_CXX`` target -- do not try
to make one wrapper call the other.

**The ``CPATH`` / ABI-mismatch trap.** If you compile against one MPI's headers
but link or run against a *different* MPI's library, the program typically
**segfaults on its first or second MPI call** (e.g. ``MPI_Comm_rank``), because
``MPI_COMM_WORLD`` is an ``int`` handle in one implementation and a pointer in the
other. A common cause: environment scripts that inject a second MPI's include
directory into ``CPATH`` (Intel oneAPI's ``setvars.sh`` does this for Intel MPI),
so the compiler silently picks up the wrong ``mpi.h`` even when CMake found the
MPI you intended. The fix is to ensure exactly one MPI stack is visible at compile
*and* run time -- clear ``CPATH`` / ``I_MPI_ROOT`` / ``LIBRARY_PATH`` from a
conflicting MPI, and check that ``ldd`` on your binary shows the same ``libmpi``
you compiled against.

**CUDA-aware MPI may not initialize -- detect, do not assume.** An MPI *built*
``--with-cuda`` is not necessarily CUDA-aware at *runtime*: the CUDA support has
to initialize on the actual machine. It can silently fail to (a notable case is
**WSL2**, whose paravirtualized CUDA driver lacks pieces such as CUDA IPC, so the
accelerator component fails ``opal_init`` even though the libraries link). If you
hard-code the device-pointer path and the runtime support is absent, MPI does a
host ``memcpy`` on a device pointer and **segfaults**. This is exactly why Step 4
queries ``MPIX_Query_cuda_support()`` at runtime and falls back to host staging --
the same binary then runs correctly on a laptop and on a GPU cluster.

**The tightly-packed contract.** ``make_datatype`` ``static_assert``\ s that
``sizeof(Jet) == ncoeffs * sizeof(Coeff)`` (no trailing padding), so a contiguous
datatype matches the array stride. This holds for every ``otinum`` shape by
construction; you do not need to check it yourself, but if you ever wrap a jet in
a larger aligned struct, describe that element with ``MPI_Type_create_resized``.

Checklist
---------

* Swap your kernel's scalar type to ``otinum`` -- no logic change.
* Define ``OTI_ENABLE_KOKKOS`` and use ``Kokkos::View<Jet*>`` for device arrays.
* Commit ``oti::mpi::make_datatype<Jet>()`` once per shape; send jets in jet units.
* For MPI + GPU, branch on ``MPIX_Query_cuda_support()`` with a host-staging
  fallback.
* Build with ``nvcc_wrapper`` as the compiler and let ``FindMPI`` supply MPI
  flags; keep a single MPI stack visible at compile and run time.
