Independent Evaluation (Gather)
===============================

The first rung of the ladder, and the simplest distributed pattern: each rank
owns a block, evaluates it independently, and the results are gathered -- **no
communication during the compute**. That keeps the focus on the one thing this
example is about: the source changes that take an ordinary ``double`` Kokkos + MPI
code and OTI-enable it so it produces derivatives. It is the integration guide
(:doc:`../integration`) applied to a concrete program.

The before/after sources are ``mpi_oti_convert/convert_before.cpp`` and
``convert_after.cpp``; the only differences between them are the changes below.

.. figure:: ../../../_static/diagrams/mpi_gather.png
   :alt: Domain split into per-rank blocks, evaluated locally, gathered to rank 0
   :width: 100%

   The distributed pattern: the grid is split into per-rank blocks; each rank
   evaluates its block independently (no communication during the compute); one
   ``MPI_Gatherv`` assembles the field on rank 0. Every grid point is a *jet* --
   the value plus its derivative coefficients -- and that jet is the committed
   ``MPI_OTINUM`` element MPI moves, with counts expressed in jets, not bytes.

The Starting Point
------------------

``convert_before.cpp`` is a plain Kokkos + MPI program: each rank evaluates a
field ``model(x,y) = sin(x)·exp(y)`` over its block of a 1000×1000 grid on the
Kokkos device, gathers the field to rank 0, and prints a sample value. It uses
``double`` throughout, ``Kokkos::View<double*>``, and ``MPI_DOUBLE``. Nothing in
it knows about OTI.

The Five Changes
----------------

.. code-block:: diff

   -using Scalar = double;
   +#include "otinum/otinum.hpp"   // 1. otinum core (+ <cmath> overloads)
   +#include "otinum/mpi.hpp"      //    optional MPI datatype helper
   +
   +using Scalar = oti::otinum<2, 2, double>;    // 2. was: using Scalar = double;

    // 0. THE KERNEL DOES NOT CHANGE -- overloaded ops carry the derivatives:
    KOKKOS_INLINE_FUNCTION Scalar model(Scalar x, Scalar y)
    {
        return sin(x) * exp(y);
    }

   -        MPI_Datatype field_type = MPI_DOUBLE;
   +        MPI_Datatype field_type = oti::mpi::make_datatype<Scalar>();   // 4.

    // inside the device kernel:
   -                Scalar x = (g / N) * h;
   -                Scalar y = (g % N) * h;
   +                Scalar x = Scalar::variable(0, (g / N) * h);   // 3. seed e_0
   +                Scalar y = Scalar::variable(1, (g % N) * h);   //    seed e_1
                    d_field(k) = model(x, y);

    // reading the result on rank 0:
   -            std::printf("sample value = %.8f\n", global[mid]);
   +            const Scalar& s = global[mid];                              // 5.
   +            std::printf("sample value = %.8f\n", s.coeff(oti::sparse({})));
   +            std::printf("        d/dx = %.8f\n", s.coeff(oti::sparse({{0,1}})));
   +            std::printf("        d/dy = %.8f\n", s.coeff(oti::sparse({{1,1}})));
   +
   +        oti::mpi::free_datatype(field_type);   // release the committed type

What each change is for:

#. **Include the optional headers.** ``otinum/otinum.hpp`` is the value type;
   ``otinum/mpi.hpp`` adds the datatype helper. Neither is needed by code that
   does not use them, so non-OTI builds are unaffected.
#. **Change the scalar type.** One ``using`` alias. Everything templated or typed
   on ``Scalar`` -- the View, the kernel, the buffers -- follows automatically.
#. **Seed the inputs as variables.** This is the one genuinely new line of
   *logic*: declaring which quantities you want derivatives with respect to.
   ``Scalar::variable(0, x0)`` is ``x0`` carrying a unit first-order
   infinitesimal in direction 0; ``variable(1, y0)`` does the same in direction 1.
#. **Describe the element to MPI.** ``oti::mpi::make_datatype<Scalar>()`` replaces
   ``MPI_DOUBLE`` -- one committed datatype for the jet -- and is freed at the end.
#. **Read the derivatives out.** The gathered value is now a jet; pull its
   coefficients with ``coeff()`` (or use it directly in further OTI arithmetic).

Note what is **absent**: the ``model()`` kernel, the decomposition, the gather,
and the control flow are byte-for-byte identical. The overloaded arithmetic and
elementary functions propagate the derivatives through unchanged code -- that is
the whole value proposition.

.. figure:: ../../../_static/diagrams/oti_jet_slices.png
   :alt: Seeded perturbation stack on the left, derivative coefficient tower on the right
   :width: 100%

   What seeding buys. On the left, each rank's block of the input is lifted into
   the OTI algebra: the domain plus the two nilpotent perturbation directions
   ``e_0`` and ``e_1`` (``Scalar::variable(0, ...)`` / ``variable(1, ...)``),
   every layer decomposed identically across ranks. Each rank then evaluates
   ``f`` on its block, and the single evaluation returns the whole output jet on
   the right -- one field per coefficient (the value and every derivative). The
   second-order coefficients were never seeded; they emerge from the algebra
   during the evaluation. Reading them back is change 5.

Build Changes
-------------

Two build-side changes, both already covered in :doc:`../integration`: define
``OTI_ENABLE_KOKKOS`` (so the jet is device-callable) and add the library include
path. In CMake:

.. code-block:: diff

    add_executable(my_app my_app.cpp)
   +target_compile_definitions(my_app PRIVATE OTI_ENABLE_KOKKOS)
   +target_include_directories(my_app PRIVATE /path/to/cpp_oti_lib/include)
    target_link_libraries(my_app PRIVATE Kokkos::kokkos MPI::MPI_CXX)

The Result
----------

Both programs run identically; the OTI version simply knows more. Run each on the
same ranks:

.. code-block:: console

   mpirun -np 4 ./build/convert_before
   mpirun -np 4 ./build/convert_after

.. code-block:: text

   # convert_before
   sample value = 0.79155923

   # convert_after
   sample value = 0.79155923
           d/dx = 1.44721739
           d/dy = 0.79155923

The value is unchanged (the OTI real part is the original computation), and the
derivatives come out alongside it -- exact, from the same distributed evaluation.
From here the full dependency model, CMake recipe, the device-pointer vs
host-staging transport choice, and the toolchain gotchas are in :doc:`../integration`.
