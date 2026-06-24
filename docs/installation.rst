Installation And Local Builds
=============================

This page collects the local commands used to build, test, and regenerate the
project documentation. The scalar C++ library is header-only, so a normal C++
program only needs a C++17 compiler and this repository's ``include``
directory. CMake, Python, Doxygen, and Kokkos are only needed for the optional
workflows below.

Use the sections in this order when setting up a new checkout:

* :ref:`installation:Getting The Code` for cloning the repository.
* :ref:`installation:Header-Only C++` for compiling a program that includes
  ``otinum/otinum.hpp``.
* :ref:`installation:Installing The CMake Package` for a relocatable
  ``find_package(otinum)`` installation.
* :ref:`installation:Focused Unit Tests` for the scalar C++ test suite.
* :ref:`installation:Python Bindings` when you want the optional Python module.
* :ref:`installation:Building These Docs` when you want to rebuild the local
  documentation site.

Getting The Code
----------------

The source code lives on GitHub at
`Samm-Py/cpp_oti_lib <https://github.com/Samm-Py/cpp_oti_lib>`_. Clone it and
move into the checkout; every command in this documentation that says "from the
repository root" means this directory:

.. code-block:: console

   git clone https://github.com/Samm-Py/cpp_oti_lib.git
   cd cpp_oti_lib

There is nothing to build at this point. The scalar C++ library is the header
tree under ``include/``, so cloning *is* the installation for the header-only
path — continue with :ref:`installation:Header-Only C++` to compile a first
program against it. The remaining prerequisites below are only needed for the
optional workflows (tests, Python bindings, Kokkos, documentation).

Prerequisites
-------------

The common local tools are:

* ``git`` for cloning the repository
* a C++17 compiler: ``g++`` 7 or newer, or ``clang++`` 5 or newer (the first
  releases with complete C++17 support); any current distribution default
  qualifies
* CMake 3.18 or newer for CTest, Python-extension, and Kokkos builds
* Python 3.9 or newer for the optional Python bindings and documentation tools

On Ubuntu and other Debian-based distributions, one command installs all of
them:

.. code-block:: console

   sudo apt-get update
   sudo apt-get install git g++ cmake python3 python3-pip python3-venv

On macOS, ``xcode-select --install`` provides the compiler and ``git``, and
`Homebrew <https://brew.sh>`_ provides the rest (``brew install cmake python``).

Confirm the versions meet the minimums above:

.. code-block:: console

   g++ --version
   cmake --version
   python3 --version

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

To verify the include path works, build the shortest complete program in the
documentation, which ships in the repository as ``examples/minimal.cpp``:

.. literalinclude:: ../examples/minimal.cpp
   :language: cpp

From inside the repository root, compile and run it like this:

.. code-block:: console

   c++ -std=c++17 -I include examples/minimal.cpp -o /tmp/oti_minimal
   /tmp/oti_minimal

Expected output is approximately:

.. code-block:: console

   f        = 4.91665
   df/dx    = 4.75182
   df/dy    = 1.35067
   d2f/dxdy = 0.704713

The exact formatting can vary slightly by standard library and compiler.

The important compiler option is ``-I include``. It points the compiler at this
repository's headers so ``#include "otinum/otinum.hpp"`` can be resolved. There
is no separate library binary to link.

For a fuller version of this example with analytic derivative checks, continue
to :doc:`tutorials/basic_usage`. For details on normalized coefficients,
multi-index access, and the low-level coefficient layout, see :doc:`api/core`.

From a separate scratch directory, use the absolute include path instead:

.. code-block:: console

   mkdir -p ~/oti_scratch
   cd ~/oti_scratch

   c++ -std=c++17 \
     -I /path/to/cpp_oti_lib/include \
     my_program.cpp \
     -o my_program

   ./my_program

In this example:

* ``my_program.cpp`` is your own source file in the scratch directory.
* ``-I /path/to/cpp_oti_lib/include`` tells the compiler where to find
  ``otinum/otinum.hpp``; replace the path with your checkout location.
* ``-o my_program`` names the executable that will be created.
* ``./my_program`` runs the executable and prints the program output.

.. note::

   The ``-I`` path must end in ``/include`` — it points at the ``include``
   directory *inside* the checkout, not at the checkout root. If the compiler
   reports ``fatal error: otinum/otinum.hpp: No such file or directory``,
   check that the path exists and ends in ``/include``:
   ``ls /path/to/cpp_oti_lib/include/otinum/otinum.hpp`` should succeed.

Installing The CMake Package
----------------------------

Install the headers and relocatable CMake package when another CMake project
should consume the library through ``find_package``:

.. code-block:: console

   cmake -S . -B build-install -DOTI_BUILD_TESTS=OFF
   cmake --install build-install --prefix /tmp/otinum-install

``-DOTI_BUILD_TESTS=OFF`` skips configuring the unit tests (on by default),
which are not needed just to install the headers and package files.

The downstream project's ``CMakeLists.txt`` can then use the exported target:

.. code-block:: cmake

   find_package(otinum CONFIG REQUIRED)
   target_link_libraries(my_target PRIVATE oti::otinum)

Configure that project with the installation prefix:

.. code-block:: console

   cmake -S /path/to/consumer -B /path/to/consumer/build \
     -DCMAKE_PREFIX_PATH=/tmp/otinum-install
   cmake --build /path/to/consumer/build

The imported target supplies the include directory and the usage requirements
recorded by the installed package. :doc:`tutorials/cmake_package` walks through
a complete consumer project, including the source files and expected output.

Focused Unit Tests
------------------

Run the focused unit tests from the repository root:

.. code-block:: console

   tests/run_unit_tests.sh

The script compiles each ``tests/test_*.cpp`` file into a separate executable,
runs each executable, and prints a pass/fail summary. By default it writes:

* test executables to ``/tmp/otinum_unit_tests``
* timestamped logs to ``logs/unit_tests_YYYYMMDD_HHMMSS.log``

Override those locations if you want all generated files outside the
repository:

.. code-block:: console

   BUILD_DIR=/tmp/my_otinum_tests LOG_DIR=/tmp/my_otinum_logs tests/run_unit_tests.sh

The same tests can also be run through CTest, CMake's test runner. Configuring
the project with CMake declares each test executable as a named CTest test, so
the usual configure/build/test sequence runs the whole suite and reports a
pass/fail summary:

.. code-block:: console

   cmake -S . -B build
   cmake --build build --parallel 2
   ctest --test-dir build --output-on-failure

``--output-on-failure`` keeps the report quiet for passing tests and prints
the full program output for any test that fails. This CTest path is what the
project's continuous integration runs; the shell script above is the quicker
loop for local development.

Each ``tests/test_*.cpp`` file covers one area of the scalar library —
construction and coefficient access, arithmetic, elementary functions,
comparisons, edge cases, ``float`` coefficients, standard-library
interoperability — and the file names state their scope, so the test list in
the runner's output doubles as a feature inventory.

One test is an exception: ``test_kokkos_smoke.cpp`` requires Kokkos, an
optional dependency not covered on this page, so both the shell runner and the
default CMake build skip it. :doc:`tutorials/kokkos_cpu` is a self-contained
guide to that path — it covers building Kokkos itself from source, configuring
``cpp_oti_lib`` against it with ``-DOTI_ENABLE_KOKKOS=ON``, and running the
smoke test; :doc:`tutorials/kokkos_gpu` does the same for the CUDA backend.

Optional Microbenchmark
-----------------------

The cost of OTI arithmetic grows quickly with the shape: an
``otinum<M, N>`` stores ``ncoeffs`` coefficients, and one multiplication
performs ``nproducts`` coefficient products, both of which grow combinatorially
in ``M`` and ``N``. The repository includes a small microbenchmark that
measures what each shape actually costs on your machine — useful when deciding
how many variables and what derivative order you can afford. It is excluded
from normal builds because benchmark timings are not correctness tests:

.. code-block:: console

   cmake -S . -B build-benchmarks \
     -DOTI_BUILD_TESTS=OFF \
     -DOTI_BUILD_BENCHMARKS=ON
   cmake --build build-benchmarks --parallel
   ./build-benchmarks/benchmark_operations

The executable prints one CSV row per shape (abbreviated here):

.. code-block:: text

   type,ncoeffs,nproducts,add_s,mul_s,div_s,exp_s,sin_s,mixed_s,sink
   oti_1_1,2,3,0.0051,0.0139,0.0251,0.0214,0.0355,0.0751,...
   oti_2_2,6,15,0.0132,0.0641,0.1003,0.0697,0.0876,0.3123,...
   oti_3_3,20,84,0.0458,0.3315,0.5354,0.4519,0.5153,1.8080,...

Reading the columns:

* ``type`` is the shape: ``oti_3_3`` is ``otinum<3, 3>``.
* ``ncoeffs`` and ``nproducts`` are the shape's storage size and the number of
  coefficient products in one multiplication — the structural reasons the
  timings differ.
* The ``*_s`` columns are wall-clock seconds for 200,000 iterations of each
  operation (add, multiply, divide, ``exp``, ``sin``, and a mixed expression).
* ``sink`` is an accumulated checksum that stops the compiler from optimizing
  the loops away; ignore its value.

Compare rows against each other rather than quoting absolute numbers: in the
run above, a ``<3, 3>`` multiplication costs about 24x a ``<1, 1>``
multiplication, roughly tracking the ``nproducts`` ratio, while addition grows
only with ``ncoeffs``. Treat the results as local comparison data rather than
portable performance claims.

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

That is the whole installation. :doc:`tutorials/python_bindings` covers using
the bound types — multi-index access, what the module exposes, the available
shapes, and how to add new instantiations.

Building These Docs
-------------------

Building the documentation needs two things: the ``doxygen`` executable (see
:ref:`installation:Prerequisites`) and the Python packages listed in
``docs/requirements.txt``. Install the Python packages with:

.. code-block:: console

   python -m pip install -r docs/requirements.txt

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

Open ``docs/_build/html/index.html`` in a browser, or serve the build locally
and browse to http://localhost:8000 (useful when the build lives on a remote
machine or in WSL):

.. code-block:: console

   python -m http.server 8000 -d docs/_build/html

Continuous Integration
----------------------

Every push and pull request runs the workflows on this page automatically: the
unit tests on both GCC and Clang, the installed-package consumer build, the
Kokkos smoke tests, the Python binding tests, and this documentation build
with warnings treated as errors, plus a coverage report published with the
hosted site. The authoritative list of jobs is the workflow file itself,
``.github/workflows/ci.yml``.
