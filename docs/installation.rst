Installation And Local Builds
=============================

This page collects the local commands used to build, test, and regenerate the
project documentation. The scalar C++ library is header-only, so a normal C++
program only needs a C++17 compiler and this repository's ``include``
directory. CMake, Python, Doxygen, and Kokkos are only needed for the optional
workflows below.

Use the sections in this order when setting up a new checkout:

* :ref:`installation:Header-Only C++` for compiling a program that includes
  ``otinum/otinum.hpp``.
* :ref:`installation:Installing The CMake Package` for a relocatable
  ``find_package(otinum)`` installation.
* :ref:`installation:Focused Unit Tests` for the scalar C++ test suite.
* :ref:`installation:Python Bindings` when you want the optional Python module.
* :ref:`installation:Building These Docs` when you want to rebuild the local
  documentation site.

Prerequisites
-------------

The common local tools are:

* a C++17 compiler such as ``g++`` or ``clang++``
* CMake 3.18 or newer for CTest, Python-extension, and Kokkos builds
* Python 3.9 or newer for the optional Python bindings and documentation tools

For documentation builds, install Doxygen before running Sphinx. On Ubuntu:

.. code-block:: console

   sudo apt-get install doxygen graphviz

The ``graphviz`` package is optional for simple page builds, but it is useful
when Doxygen diagrams are enabled.

Header-Only C++
---------------

The core C++ library does not require an install step and has no separate
library binary to build. You can compile your own source file directly against
this repository's ``include`` directory. The OTI implementation is compiled as
part of your program through those headers.

From inside the repository root, compile the minimal example like this:

.. code-block:: console

   c++ -std=c++17 -I include examples/minimal.cpp -o /tmp/oti_minimal

That command only builds the executable. Run it separately:

.. code-block:: console

   /tmp/oti_minimal

The expected output is shown in :doc:`readme`.

From a separate scratch directory, use the absolute include path instead:

.. code-block:: console

   mkdir -p ~/oti_scratch
   cd ~/oti_scratch

   c++ -std=c++17 \
     -I /path/to/cpp_oti_lib/include \
     basic_usage.cpp \
     -o basic_usage

   ./basic_usage

In this example:

* ``basic_usage.cpp`` is your local source file in the scratch directory.
* ``-I /path/to/cpp_oti_lib/include`` tells the compiler where to find
  ``otinum/otinum.hpp``; replace the path with your checkout location.
* ``-o basic_usage`` names the executable that will be created.
* ``./basic_usage`` runs the executable and prints the program output.

Installing The CMake Package
----------------------------

Install the headers and relocatable CMake package when another CMake project
should consume OTIlib through ``find_package``:

.. code-block:: console

   cmake -S . -B build-install \
     -DOTI_BUILD_TESTS=OFF \
     -DOTI_BUILD_PYTHON=OFF
   cmake --install build-install --prefix /tmp/otinum-install

The downstream project's ``CMakeLists.txt`` can then use the exported target:

.. code-block:: cmake

   find_package(otinum CONFIG REQUIRED)
   target_link_libraries(my_target PRIVATE oti::otinum)

Configure that project with the installation prefix:

.. code-block:: console

   cmake -S /path/to/consumer -B /path/to/consumer/build \
     -DCMAKE_PREFIX_PATH=/tmp/otinum-install
   cmake --build /path/to/consumer/build

The imported target supplies the include directory and C++17 requirement.
Kokkos-enabled installations also preserve their ``Kokkos::kokkos`` dependency.
Set ``-DOTI_INSTALL_CMAKE_PACKAGE=OFF`` when configuring OTIlib if the install
rules are not needed, as in Python wheel builds.

Focused Unit Tests
------------------

Run the focused unit tests from the repository root:

.. code-block:: console

   tests/run_unit_tests.sh

The script compiles each ``tests/test_*.cpp`` file into a separate executable,
runs each executable, and prints a pass/fail summary. By default it writes:

* test executables to ``/tmp/otinum_unit_tests``
* timestamped logs to ``logs/unit_tests_YYYYMMDD_HHMMSS.log``

The focused tests are also registered with CTest by default:

.. code-block:: console

   cmake -S . -B build
   cmake --build build --parallel 2
   ctest --test-dir build --output-on-failure

Set ``-DOTI_BUILD_TESTS=OFF`` when you only want the interface target.

The focused tests cover the main scalar library behavior:

* ``test_layout_tables.cpp`` checks multi-index ranking, graded coefficient
  order, factorial metadata, and product-table structure.
* ``test_construction_access.cpp`` checks zero/default construction, scalar
  construction, variable seeding, direct coefficient construction, coefficient
  setters, and out-of-order multi-index access.
* ``test_linear_arithmetic.cpp`` checks addition, subtraction, negation, and
  scalar/OTI mixed arithmetic.
* ``test_multiplication_division.cpp`` checks polynomial multiplication,
  inverse expansion, OTI division, and scalar division.
* ``test_truncated_operations.cpp`` checks order-limited addition and
  multiplication.
* ``test_exp_log_pow.cpp`` checks exponential, logarithm, scalar powers,
  square roots, cube roots, and OTI-valued powers.
* ``test_trig_hyperbolic.cpp`` checks trigonometric and hyperbolic functions.
* ``test_abs_large_shapes.cpp`` checks absolute value behavior and selected
  larger ``(M, N)`` shapes.
* ``test_comparisons.cpp`` checks OTI/OTI and mixed scalar comparisons.
* ``test_edge_cases.cpp`` checks constant-only shapes, truncation boundaries,
  sparse-index validation, NaN propagation, and invalid table lookups.
* ``test_float_coefficients.cpp`` checks the ``otinum<M, N, float>`` path,
  including float storage and arithmetic with scalar literals.
* ``test_fused_ops.cpp`` checks fused multiply-add style helpers.
* ``test_interop.cpp`` checks standard-library and mixed-type interoperability.
* ``test_profile.cpp`` checks the optional profiling counters and CSV output
  helpers.

The direct shell runner skips ``test_kokkos_smoke.cpp`` by default. Run the
Kokkos smoke test through CMake with ``-DOTI_ENABLE_KOKKOS=ON`` so the build can
find and link ``Kokkos::kokkos`` correctly; the CPU and GPU Kokkos tutorials
show those workflows.

Override those locations if you want all generated files outside the repository:

.. code-block:: console

   BUILD_DIR=/tmp/my_otinum_tests LOG_DIR=/tmp/my_otinum_logs tests/run_unit_tests.sh

Optional Microbenchmark
-----------------------

The repository includes a small operations microbenchmark. It is excluded from
normal builds because benchmark timings are not correctness tests:

.. code-block:: console

   cmake -S . -B build-benchmarks \
     -DOTI_BUILD_TESTS=OFF \
     -DOTI_BUILD_BENCHMARKS=ON
   cmake --build build-benchmarks --parallel
   ./build-benchmarks/benchmark_operations

The executable prints CSV rows for several ``(M, N)`` shapes. Treat the results
as local comparison data rather than portable performance claims.

Python Bindings
---------------

Python bindings are optional and are not required for the header-only C++
library. Install them in editable mode from the repository root. This builds
the extension module and makes it importable in the active Python environment:

.. code-block:: console

   python -m pip install -e .

The extension exposes a fixed set of template instantiations such as
``OTI_1_3``, ``OTI_2_3``, and ``OTI_3_3``.

After installation, test the import:

.. code-block:: console

   python -c "import otinum as oti; print(oti.OTI_2_3)"

Run the Python binding smoke tests with ``pytest``:

.. code-block:: console

   python -m pip install -e ".[test]"
   pytest tests/test_python_bindings.py

Using The Bound Types
~~~~~~~~~~~~~~~~~~~~~

Each Python class corresponds to one concrete C++ template instantiation:

.. code-block:: python

   import otinum as oti

   T = oti.OTI_2_3
   x = T.variable(0, 1.5)
   y = T.variable(1, 0.3)
   f = oti.sin(x * y) + oti.exp(x)

   print(f.real())
   print(f.partial([1, 0]))
   print(f.data())

The multi-index arguments passed to ``coeff``, ``partial``, ``set_coeff``, and
``set_partial`` are Python lists or tuples. Their length must match the number
of variables in the bound type. For ``OTI_2_3``, use two entries such as
``[1, 0]``. For ``OTI_3_3``, use three entries such as ``[0, 1, 1]``.

For high-dimensional types, the same methods also accept sparse multi-indices
as ``[variable_index, derivative_order]`` pairs:

.. code-block:: python

   T = oti.OTI_10_2
   x0 = T.variable(0, 1.5)
   x7 = T.variable(7, 2.0)
   f = x0 * x7

   print(f.partial([[0, 1]]))          # df/dx0
   print(f.partial([[7, 1]]))          # df/dx7
   print(f.partial([[0, 1], [7, 1]])) # d2f/dx0 dx7

Variable indices are zero-based, matching ``T.variable(index, value)``. Sparse
pairs with the same variable are added, so ``[[7, 1], [7, 1]]`` requests the
second derivative with respect to variable ``7``. Dense multi-indices remain
the clearest form for small ``M``; sparse multi-indices are mainly for types
such as ``OTI_10_2`` or larger. ``OTI_10_2`` is an example of a shape you can
add using the instantiation steps below.

Currently Bound Instantiations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The binding file registers this small grid by default:

.. code-block:: text

   OTI_1_1  OTI_1_2  OTI_1_3
   OTI_2_1  OTI_2_2  OTI_2_3
   OTI_3_1  OTI_3_2  OTI_3_3

Python cannot instantiate arbitrary C++ templates at runtime. A type such as
``oti::otinum<10, 1>`` must be explicitly compiled into the extension before it
can be imported as a Python class.

Adding A New Instantiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

To add ``OTI_10_1``, edit ``bindings/python/otinum_py.cpp``. At the bottom of
the file, inside ``PYBIND11_MODULE(otinum, m)``, add:

.. code-block:: cpp

   bind_otinum<10, 1>(m, "OTI_10_1");

The registration block would then include:

.. code-block:: cpp

   PYBIND11_MODULE(otinum, m)
   {
       m.doc() = "Python bindings for static OTI numbers";

       bind_otinum<1, 1>(m, "OTI_1_1");
       bind_otinum<1, 2>(m, "OTI_1_2");
       bind_otinum<1, 3>(m, "OTI_1_3");

       // ...

       bind_otinum<10, 1>(m, "OTI_10_1");
   }

Rebuild the editable install after changing the bindings:

.. code-block:: console

   python -m pip install -e . --force-reinstall --no-build-isolation

Then test the new type:

.. code-block:: console

   python -c "import otinum as oti; print(oti.OTI_10_1.nvars, oti.OTI_10_1.order, oti.OTI_10_1.ncoeffs)"

For ``OTI_10_1``, ``ncoeffs`` is ``11`` because the type stores the real
coefficient plus one first-order coefficient for each of the ten variables.

Practical Notes
~~~~~~~~~~~~~~~

Adding more instantiations increases compile time and extension size because
each ``otinum<M, N>`` type is a distinct C++ template instantiation. Prefer
binding the shapes you actually use rather than a very large grid. Large
``M``/``N`` combinations can also create many coefficients per value, so check
``T.ncoeffs`` before adding broad Python coverage.

Building These Docs
-------------------

Build the documentation from the repository root. The generated C++ API
reference requires the ``doxygen`` executable in addition to the Python
packages.

Install the Python documentation dependencies into the active Python
environment:

.. code-block:: console

   python -m pip install -r docs/requirements.txt

Alternatively, install the package's optional documentation dependencies:

.. code-block:: console

   python -m pip install -e ".[docs]"

Generate the Doxygen XML, then run Sphinx:

.. code-block:: console

   doxygen docs/Doxyfile
   sphinx-build -E -a -W -b html docs docs/_build/html

The command above:

* writes generated C++ API XML to ``docs/api/xml/``
* reads source files from ``docs/``
* writes generated HTML to ``docs/_build/html/``
* uses ``-E -a`` to force a complete rebuild
* uses ``-W`` so warnings are treated as errors

Open the generated landing page in a browser:

.. code-block:: text

   docs/_build/html/index.html

Continuous Integration
----------------------

The GitHub Actions workflow in ``.github/workflows/ci.yml`` runs these checks:

* the focused C++ tests with both GCC and Clang through CMake/CTest and
  ``tests/run_unit_tests.sh``
* installation followed by an external ``find_package(otinum)`` consumer build
* the Kokkos CPU/OpenMP smoke test
* the Kokkos GPU/CUDA smoke test when a CUDA device and ``nvcc`` are available
* the Python binding smoke tests in ``tests/test_python_bindings.py``
* the Doxygen XML generation and Sphinx documentation build with warnings as
  errors
* the generated coverage report, published with the documentation site on
  pushes to ``master``; this report includes both the scalar focused tests and
  the Kokkos OpenMP smoke test

By default, the GPU job runs on ``ubuntu-latest`` and skips when no CUDA device
is visible. Set the repository variable ``KOKKOS_GPU_RUNNER`` to a CUDA-capable
runner label to make that job exercise an available GPU.
