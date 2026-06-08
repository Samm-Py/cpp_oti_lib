Full Generated Index
====================

The focused API pages render the Doxygen XML without duplicating declarations
across the Sphinx build. The complete generated XML remains available on disk
for tools and future automation:

.. code-block:: text

   docs/api/xml/index.xml

The generated XML includes all headers under ``include/otinum``, including the
``detail`` namespace. Use the Sphinx search box for rendered documentation, or
inspect the XML directly when debugging Doxygen/Breathe coverage.
