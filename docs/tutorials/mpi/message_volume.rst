Message Volume
==============

Counting in *jets* keeps the API clean, but do not lose sight of what one jet
costs on the wire: a jet is ``ncoeffs`` coefficients, so **every message is**
``ncoeffs`` **times the size of the equivalent scalar message**. The coefficient
count grows combinatorially with the number of seeded directions :math:`M` and the
derivative order :math:`N`, as :math:`\binom{M+N}{N}`:

.. list-table::
   :header-rows: 1

   * - Algebra
     - Carries
     - ``ncoeffs``
     - Bytes (``double``) vs scalar
   * - ``otinum<2,1>``
     - value + gradient
     - 3
     - 24 B (3×)
   * - ``otinum<2,2>``
     - value + gradient + Hessian
     - 6
     - 48 B (6×)
   * - ``otinum<3,2>``
     - 3 directions, 2nd order
     - 10
     - 80 B (10×)
   * - ``otinum<3,3>``
     - 3 directions, 3rd order
     - 20
     - 160 B (20×)

This is a multiplier on **communication**, not just storage: a halo exchange, a
gather, or an ``Allreduce`` moves ``ncoeffs`` times the bytes a plain-``double``
solve would, and bandwidth-bound exchanges scale accordingly. The practical rule
is to **seed only the directions you need and use the lowest derivative order that
answers the question** -- the jet shape, not the rank count, sets the volume.
