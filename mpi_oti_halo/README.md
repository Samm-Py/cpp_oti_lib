# mpi_oti_halo — halo exchange (Jacobi), the first communicating solver

A distributed steady-state heat solve (Laplace's equation, 5-point Jacobi
stencil) over a 2D Cartesian grid of MPI ranks. Unlike the embarrassingly-parallel
gather toy in [`../mpi_oti_toy`](../mpi_oti_toy), this one **communicates every
iteration**: each rank exchanges a one-cell ghost layer with its four neighbours,
then sweeps its interior.

## The OTI angle

The two hot walls carry their temperature as **seeded variables** (West = `e_0`,
South = `e_1`), so every converged cell drags along `d(temperature)/dT_west` and
`d(temperature)/dT_south` — the parameter sensitivity of the whole field, from a
single solve. The jet is `oti::otinum<2, 1, double>` (value + two first
derivatives).

Because the problem is linear and both walls sit at 1.0, the centre cell satisfies
`temperature == d/dT_west + d/dT_south`, and by diagonal symmetry the two
sensitivities are equal — a free self-consistency check on the output.

The executable also performs an independent centered finite-difference check.
It repeats the serial problem with plain `double` arithmetic using finite-
difference step size `h = 1e-6`, at `T_west ± h` and `T_south ± h`, then
compares those slopes with the OTI coefficients at the centre. The check uses an
absolute tolerance of `1e-8` and contributes to the program's exit status.

To export the full-grid absolute errors and plot them:

```sh
mpirun -np 4 ./mpi_oti_halo --fd-error-csv fd_error.csv
python3 plot_fd_error.py fd_error.csv mpi_halo_fd_error.png
```

## The MPI-specific surface

Two derived datatypes, both built on the committed jet element
(`oti::mpi::make_datatype<Jet>()`):

| Halo | Direction | Memory | Datatype |
|------|-----------|--------|----------|
| Row (N/S) | along contiguous axis | packed | `count = ny` of `MPI_OTINUM` |
| Column (E/W) | strided across rows | gappy | `MPI_Type_vector(nx, 1, stride, MPI_OTINUM)` |

If the strided column type were wrong, the columns would corrupt and the result
would diverge from the serial reference — so the bit-exact check is an end-to-end
test of both halos.

## Verify without a gather

Every rank redundantly runs the identical serial Jacobi on the full domain and
compares its own tile against the matching subblock. Jacobi is deterministic and
the stencil arithmetic is shared verbatim between the serial and distributed
paths, so the distributed result is **bit-for-bit identical** to serial —
including all derivative coefficients. The only real communication is the halo
exchange; verification needs no `MPI_Gatherv`.

## Build & run

```sh
mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_halo
mpirun -np 4 ./mpi_oti_halo
```

`MPI_Dims_create` factors the rank count into a 2D process grid: 4 → 2×2, 6 →
3×2, 9 → 3×3 (these exercise the strided column halo); primes give a 1D strip.
Verified bit-exact at `np = 1, 2, 3, 4, 6, 7, 9`. The program returns nonzero on
any mismatch, so it works as a CI gate.
