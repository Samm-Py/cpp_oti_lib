MPI
===

These tutorials demonstrate how OTI numbers are distributed across MPI ranks.
They rely on a single fact about the layout: an ``oti::otinum<M, N, Coeff>`` is a
fixed-size, contiguous block of ``ncoeffs`` ``Coeff`` values (``float`` or
``double``). Therefore, an array of jets is one packed buffer, and a
single committed ``MPI_Datatype`` describes one jet -- so MPI moves jets as a
first-class element, with send/receive/collective counts expressed in *jets*
rather than bytes. (See :doc:`../soa_layout` for the alignment rule that keeps the
layout padding-free.)

``cpp_oti_lib`` provides MPI helpers for communicating OTI numbers.
:doc:`datatype_helper` describes how the jet is committed as a datatype
to MPI. :doc:`message_volume` covers the cost: each message is ``ncoeffs`` × the
equivalent scalar message. :doc:`Converting Code to OTI <converting/index>`
shows how to use OTI numbers in an MPI code, including custom
``MPI_Op``\ s for reductions and derived datatypes for communication across
ranks. The separate :doc:`../integration` tutorial is the broader culmination
-- bringing this together with Kokkos for a full MPI + GPU application.

.. toctree::
   :hidden:
   :maxdepth: 1

   datatype_helper
   message_volume
   converting/index
