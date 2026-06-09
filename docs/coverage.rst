Coverage Report
===============

Coverage is generated outside Sphinx and copied into the documentation build as
a static HTML report. The recommended tool is ``gcovr``.

Generate Coverage
-----------------

From a clean working tree, build and run the focused tests with GCC coverage
flags:

.. code-block:: console

   BUILD_DIR=/tmp/otinum_coverage_tests \
   LOG_DIR=/tmp/otinum_coverage_logs \
   CXXFLAGS="-std=c++17 -O0 -g --coverage -Wall -Wextra -pedantic" \
   tests/run_unit_tests.sh

Optionally, build and run the Kokkos OpenMP smoke test with coverage flags.
This is the path used by CI so the generated report includes the
``OTI_ENABLE_KOKKOS`` implementation:

.. code-block:: console

   git clone --depth 1 https://github.com/kokkos/kokkos.git /tmp/kokkos
   cmake -S /tmp/kokkos -B /tmp/kokkos-build-openmp-coverage \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-install-openmp-coverage \
     -DCMAKE_CXX_COMPILER=g++ \
     -DKokkos_ENABLE_OPENMP=ON \
     -DKokkos_ARCH_NATIVE=ON
   cmake --build /tmp/kokkos-build-openmp-coverage --target install --parallel

   cmake -S . -B build-kokkos-openmp-coverage \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-openmp-coverage \
     -DCMAKE_CXX_FLAGS="-O0 -g --coverage" \
     -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_PYTHON=OFF \
     -DOTI_BUILD_TESTS=OFF
   cmake --build build-kokkos-openmp-coverage --parallel
   ctest --test-dir build-kokkos-openmp-coverage --output-on-failure

Then generate the report into the same path used by the hosted documentation:
if you skipped the Kokkos step, omit ``build-kokkos-openmp-coverage`` from the
final command.

.. code-block:: console

   mkdir -p docs/_build/html/generated/coverage
   GCOV=${GCOV:-gcov}
   gcovr \
     --root . \
     --gcov-executable "$GCOV" \
     --filter include/otinum/ \
     --html-details docs/_build/html/generated/coverage/index.html \
     --xml docs/_build/html/generated/coverage/coverage.xml \
     /tmp/otinum_coverage_tests \
     build-kokkos-openmp-coverage

If your compiler and default ``gcov`` come from different GCC versions, set
``GCOV`` explicitly, for example ``GCOV=gcov-13``. The report output lives under
``docs/_build/html/generated/coverage/``. That directory is ignored by Git
because it is generated.

CUDA/GPU Kokkos is tested separately when a device is available. It is not
merged into the GCC coverage report because CUDA device coverage requires
different compiler/tooling support from the OpenMP-backed coverage build.

Clean Coverage Artifacts
------------------------

The generated ``*.gcda``, ``*.gcno``, and ``*.gcov`` files are ignored by Git,
but they can still make a local checkout noisy after coverage experiments. To
remove them:

.. code-block:: console

   find . \( -name '*.gcda' -o -name '*.gcno' -o -name '*.gcov' \) -type f -delete

The scalar coverage command above writes its build products under ``/tmp``. The
Kokkos coverage command writes coverage side files under the ignored
``build-kokkos-openmp-coverage`` directory.

View The Report
---------------

After running the commands above, open:

.. code-block:: text

   docs/_build/html/generated/coverage/index.html

After the CI documentation deployment has completed on ``master`` and the docs
site is available, the hosted documentation includes:

.. raw:: html

   <p><a href="generated/coverage/index.html">generated/coverage/index.html</a></p>

CI Integration
--------------

The repository CI runs the focused C++ tests, Python binding tests, Kokkos CPU
smoke test, optional Kokkos GPU smoke test, and docs build. The docs job also
generates this coverage report before publishing the GitHub Pages artifact. The
coverage report includes the scalar focused tests and the Kokkos OpenMP smoke
test so both ordinary and ``OTI_ENABLE_KOKKOS`` header paths are exercised. The
important constraint is to keep coverage outputs generated, not committed.
