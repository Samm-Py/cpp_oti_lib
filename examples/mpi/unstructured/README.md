# mpi_oti_unstructured — unstructured ghost exchange (`MPI_Type_indexed`)

The fourth rung of the MPI ladder. The halo rung
([`../halo`](../halo)) communicated over a **structured** grid,
where a neighbour's ghost data is contiguous (a row) or strided (a column) —
which `MPI_Type_vector` describes. Real unstructured meshes have no such
regularity: the nodes another rank needs from me are an **arbitrary, scattered
subset** of my owned nodes. That irregular index list is exactly what
`MPI_Type_indexed` is for.

## The problem

Steady-state diffusion on an unstructured graph — a ring plus a fixed set of
deterministic "chords" (a small-world graph). Each non-source node relaxes to the
degree-weighted average of its graph neighbours: the graph analogue of the 5-point
Jacobi heat solve in the halo rung. Two fixed source nodes hold the boundary
values.

## The OTI angle

The two sources carry their value as **seeded variables** (`A = e_0`, `B = e_1`),
so every converged node drags along `d(value)/dA` and `d(value)/dB` — the source
sensitivity of the whole field, from a single solve. The jet is
`oti::otinum<2, 1, double>` (value + two first derivatives).

Because the map is linear and both sources sit at 1.0, every node satisfies
`value == d/dA + d/dB` — a free self-consistency check on the output.

## The MPI-specific surface

One derived datatype **per neighbour**, built on the committed jet element
(`oti::mpi::make_datatype<Jet>()`):

| Direction | Memory | Datatype |
|-----------|--------|----------|
| **Receive** ghosts | grouped by owner → contiguous | `count` of `MPI_OTINUM` |
| **Send** owned nodes a neighbour needs | arbitrary, scattered | `MPI_Type_indexed(MPI_OTINUM)`, block lengths all 1 |

Ghosts are laid out grouped by owning rank, so each neighbour's incoming block is
a plain contiguous count. The owned nodes a neighbour needs are scattered through
the local array, so the send side uses `MPI_Type_indexed` to gather them in place
with no manual packing — the unstructured analogue of the strided column type in
the halo rung. (When every block is length 1, as here,
`MPI_Type_create_indexed_block` is the shorter spelling.)

## Verify without a gather

Every rank redundantly runs the identical serial Jacobi on the full graph and
compares its owned nodes against the matching global entries. The sweep arithmetic
(neighbours accumulated in global-id order) is shared verbatim between the serial
and distributed paths, so the distributed result is **bit-for-bit identical** to
serial — including all derivative coefficients. The only real communication is the
ghost exchange; verification needs no gather.

The executable also performs an independent centered finite-difference check of
the two source sensitivities over the whole graph (step `h = 1e-6`, absolute
tolerance `1e-8`), contributing to the exit status.

## Build & run

```sh
mpicxx -std=c++17 -O2 -I ../../../include main.cpp -o mpi_oti_unstructured
mpirun -np 4 ./mpi_oti_unstructured
```

The contiguous block partition works for any rank count; the cross-partition
chords make the ghost lists irregular regardless of how the nodes are split.
Verified bit-exact and rank-count-invariant at `np = 1, 2, 3, 4, 5, 7, 8`. The
program returns nonzero on any mismatch or failed finite-difference comparison, so
it works as a CI gate.
