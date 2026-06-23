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

The OTI-specific surface is small: :doc:`datatype_helper` is the one helper you
need (a committed ``MPI_Datatype`` per jet shape), and :doc:`message_volume` is the
one cost to keep in mind (each message is ``ncoeffs`` × the equivalent scalar
message). :doc:`Converting Code to OTI <converting/index>` is the ladder of
before/after examples that put them to work, and the separate
:doc:`../integration` tutorial is the broader culmination -- bringing this together
with Kokkos for a full MPI + GPU application.

The example sources live at the repository root in ``mpi_oti_convert/``
(conversion before/after, the first rung), ``mpi_oti_reduce/`` (global reduction
with a custom ``MPI_Op``), ``mpi_oti_halo/`` (the Jacobi halo-exchange solver),
``mpi_oti_unstructured/`` (irregular ghost lists via ``MPI_Type_indexed``), and
``mpi_oti_toy/`` (the datatype/accuracy/scaling verification harness); the
optional GPU sources in ``mpi_oti_gpu_toy/`` are covered by the
:doc:`../integration` tutorial.

.. toctree::
   :hidden:
   :maxdepth: 1

   datatype_helper
   message_volume
   converting/index
