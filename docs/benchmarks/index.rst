Benchmarks
==========

These pages collect repeatable performance studies for ``cpp_oti_lib``. They run
on CUDA for the GPU numbers, and on a Serial/OpenMP Kokkos for CPU runs.

.. toctree::
   :maxdepth: 1

   gpu_optimization_workflow
   heat_equation

Before you start
----------------

The benchmarks need Kokkos. For the GPU numbers you need a CUDA-enabled Kokkos
build; for CPU runs a Serial/OpenMP build is enough. If you do not already have
one, the :doc:`../tutorials/kokkos_gpu` and :doc:`../tutorials/kokkos_cpu`
tutorials walk through it. You also need CMake 3.16+, a C++17 compiler, and
Python 3 with ``matplotlib`` for the runners and plotters.

Two things commonly trip up a first CUDA build:

* **Host compiler.** CUDA Kokkos needs C++20 on the NVCC host pass, so
  ``nvcc_wrapper``'s host compiler must be **g++ 11 or newer** (g++ 10 emits
  ``-std=c++2a``, which NVCC rejects). Select it with
  ``export NVCC_WRAPPER_DEFAULT_COMPILER=g++-11``.
* **No GPU?** The benchmarks still build and run under a Serial/OpenMP Kokkos.
  ``run_benchmarks.py`` expects a CUDA backend by default; pass
  ``--allow-non-cuda`` to collect CPU results instead.

Then configure the library with Kokkos and the benchmark targets enabled:

.. code-block:: console

   cmake -S . -B build-cuda \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
     -DCMAKE_PREFIX_PATH=/path/to/kokkos-cuda-install \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_BENCHMARKS=ON

Each page below then has its own collect-and-plot commands.
