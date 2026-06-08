Coverage Report
===============

Coverage is generated outside Sphinx and linked into the documentation as a
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

Then generate the report:

.. code-block:: console

   mkdir -p docs/generated/coverage
   gcovr \
     --root . \
     --filter include \
     --html-details docs/generated/coverage/index.html \
     --xml docs/generated/coverage/coverage.xml

The report output lives under ``docs/generated/coverage/``. That directory is
ignored by Git because it is generated.

View The Report
---------------

After running the commands above, open:

.. code-block:: text

   docs/generated/coverage/index.html

CI Integration
--------------

A future CI workflow can run the same commands and publish
``docs/generated/coverage`` as an artifact or as part of a documentation site.
The important constraint is to keep coverage outputs generated, not committed.
