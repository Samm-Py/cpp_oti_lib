Coverage Report
===============

Coverage is generated outside Sphinx and hosted with the documentation as a
static HTML report. The recommended tool is ``gcovr``.

Generate Coverage
-----------------

From a clean working tree, build and run the focused tests with GCC coverage
flags:

.. code-block:: console

   BUILD_DIR=/tmp/otinum_coverage_tests \
   LOG_DIR=/tmp/otinum_coverage_logs \
   CXXFLAGS="-std=c++17 -O0 -g --coverage -Wall -Wextra -pedantic" \
   tests/run_unit_tests.sh

Then generate the report into the same path used by the hosted documentation:

.. code-block:: console

   mkdir -p docs/_build/html/generated/coverage
   GCOV=${GCOV:-gcov}
   gcovr \
     --root . \
     --object-directory /tmp/otinum_coverage_tests \
     --gcov-executable "$GCOV" \
     --filter include \
     --html-details docs/_build/html/generated/coverage/index.html \
     --xml docs/_build/html/generated/coverage/coverage.xml

If your compiler and default ``gcov`` come from different GCC versions, set
``GCOV`` explicitly, for example ``GCOV=gcov-13``. The report output lives under
``docs/_build/html/generated/coverage/``. That directory is ignored by Git
because it is generated.

View The Report
---------------

After running the commands above, open:

.. code-block:: text

   docs/_build/html/generated/coverage/index.html

On the hosted documentation site, open:

.. raw:: html

   <p><a href="generated/coverage/index.html">generated/coverage/index.html</a></p>

CI Integration
--------------

The repository CI runs the focused C++ tests, Python binding tests, Kokkos CPU
smoke test, optional Kokkos GPU smoke test, and docs build. The docs job also
generates this coverage report before publishing the GitHub Pages artifact. The
important constraint is to keep coverage outputs generated, not committed.
