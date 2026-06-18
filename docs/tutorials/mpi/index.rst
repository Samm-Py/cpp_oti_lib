MPI
===

These tutorials distribute OTI numbers across MPI ranks. They rely on a single
fact about the layout: an ``oti::otinum<M, N, Coeff>`` is a fixed-size,
contiguous block of ``ncoeffs`` coefficients with no pointers, no heap, and no
trailing padding (see :doc:`../soa_layout` for the alignment rule that keeps it
padding-free). An array of jets is therefore one packed buffer, and a single
committed ``MPI_Datatype`` describes one jet -- so MPI moves jets as a
first-class element, with send/receive/collective counts expressed in *jets*
rather than bytes. There is no serialization layer to write.

The optional header ``otinum/mpi.hpp`` provides that datatype through
``oti::mpi::make_datatype<T>()``. It is deliberately **not** part of the
``otinum.hpp`` umbrella, so the core headers stay MPI-free and non-MPI builds
carry no dependency.

The section is organized by what you need to pull in. **Getting Started** is the
hardware-independent foundation: the datatype, and a check that distributing
never changes the answer. **Across CPU Ranks** and **Across GPUs** are mirror
images of the same idea -- distribute the evaluation and run it -- on each kind of
hardware. **Converting Code to OTI** is a growing ladder of before/after examples,
ordered by communication complexity. **Reference** is the full integration guide.

The example sources live at the repository root in ``mpi_oti_toy/`` (CPU),
``mpi_oti_gpu_toy/`` (GPU), and ``mpi_oti_convert/`` (conversion before/after).

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   make_datatype
   accuracy

.. toctree::
   :maxdepth: 1
   :caption: MPI on CPU

   cpu

.. toctree::
   :maxdepth: 1
   :caption: MPI on GPU

   gpu

.. toctree::
   :maxdepth: 1
   :caption: Converting Code to OTI

   converting/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   integration
