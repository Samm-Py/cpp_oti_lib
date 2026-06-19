#ifndef OTINUM_MPI_HPP
#define OTINUM_MPI_HPP

// Optional MPI interop for cpp_oti_lib.
//
// Include this ONLY in translation units that already pull in <mpi.h> and link
// against MPI. It is deliberately NOT part of the otinum.hpp umbrella, so the
// core headers stay MPI-free and non-MPI users carry no dependency.
//
// Why this is so thin: otinum<M,N,Coeff> is a fixed-size contiguous block of
// ncoeffs coefficients with no pointers, no heap, and (by the otinum_alignment
// rule in core.hpp) no trailing padding. So an array of otinums is one packed
// buffer, and a single committed MPI_Datatype describes one jet -- there is no
// serialization layer to write. This header just builds and commits that type.

#include <mpi.h>

#include <cstddef>
#include <type_traits>

#include "otinum/core.hpp"

namespace oti {
namespace mpi {

// --- coefficient type -> MPI predefined datatype --------------------------
// Only float and double are mapped; any other coefficient type is a hard
// compile error (a clear stop beats silently picking the wrong datatype).
template <class Coeff>
struct mpi_scalar {
    static_assert(sizeof(Coeff) == 0,
                  "oti::mpi: no MPI datatype mapping for this coefficient type "
                  "(only float and double are provided)");
};

template <>
struct mpi_scalar<float> {
    static MPI_Datatype value() noexcept { return MPI_FLOAT; }
};

template <>
struct mpi_scalar<double> {
    static MPI_Datatype value() noexcept { return MPI_DOUBLE; }
};

// --- tightly-packed contract ----------------------------------------------
// True iff T's object representation is exactly its coefficient block, i.e.
// sizeof(T) == ncoeffs * sizeof(Coeff) with no trailing padding. This holds for
// every otinum shape by construction (otinum_alignment only ever promotes to an
// alignment the byte size is already a multiple of), but is checked so that a
// future layout change cannot silently desync a raw-buffer / MPI transfer.
template <class T>
struct is_tightly_packed : std::false_type {};

template <int M, int N, class Coeff>
struct is_tightly_packed<otinum<M, N, Coeff>>
    : std::bool_constant<sizeof(otinum<M, N, Coeff>) ==
                         static_cast<std::size_t>(otinum<M, N, Coeff>::ncoeffs) *
                             sizeof(Coeff)> {};

template <class T>
inline constexpr bool is_tightly_packed_v = is_tightly_packed<T>::value;

// --- the datatype helper ---------------------------------------------------
// Build AND commit a contiguous MPI_Datatype describing one otinum<M,N,Coeff>:
// ncoeffs contiguous Coeff scalars. The caller OWNS the returned handle and must
// release it with free_datatype (or MPI_Type_free). Commit once per (M,N,Coeff)
// at setup and reuse it for every send/recv/collective; it is also the base
// element for derived types (MPI_Type_vector for strided halos, MPI_Type_indexed
// for unstructured ghost-node lists).
template <class T>
[[nodiscard]] MPI_Datatype make_datatype()
{
    static_assert(is_tightly_packed_v<T>,
                  "otinum has trailing padding; a contiguous MPI datatype would "
                  "desync from the array stride -- describe the element with "
                  "MPI_Type_create_resized (extent = sizeof(T)) instead.");
    MPI_Datatype dt;
    MPI_Type_contiguous(T::ncoeffs, mpi_scalar<typename T::coeff_type>::value(),
                        &dt);
    MPI_Type_commit(&dt);
    return dt;
}

// Free a datatype obtained from make_datatype, so call sites read make/free
// symmetrically. Sets the handle to MPI_DATATYPE_NULL.
inline void free_datatype(MPI_Datatype& dt) noexcept { MPI_Type_free(&dt); }

// --- custom reduction operators --------------------------------------------
// MPI can MPI_SUM an int or a double, but it has no idea how to combine an
// otinum, so reductions (MPI_Reduce / MPI_Allreduce) over MPI_OTINUM need a user
// MPI_Op. sum_fn is that operator: it adds two buffers of jets coefficient-wise
// (otinum::operator+=). It matches the MPI_User_function signature and is what
// make_sum_op registers; you rarely need to name it directly.
template <class T>
void sum_fn(void* in, void* inout, int* len, MPI_Datatype* /*dt*/)
{
    const T* a = static_cast<const T*>(in);
    T* b = static_cast<T*>(inout);
    const int n = *len;
    for (int i = 0; i < n; ++i) b[i] += a[i];
}

// Build AND commit an MPI_Op that sums jets, so a reduction over MPI_OTINUM
// yields a quantity of interest carrying its gradient/Hessian -- without the
// caller hand-rolling the user function. Commutative (addition is). The caller
// OWNS the returned handle and releases it with free_op (or MPI_Op_free).
//
// (For a sum, jet addition is coefficient-wise, so this equals an MPI_SUM over
// the ncoeffs raw scalars; the custom op generalizes to any associative jet
// combine and keeps the reduction expressed in MPI_OTINUM units.)
template <class T>
[[nodiscard]] MPI_Op make_sum_op()
{
    MPI_Op op;
    MPI_Op_create(&sum_fn<T>, /*commute=*/1, &op);
    return op;
}

// Free an MPI_Op obtained from make_sum_op, symmetric with make/free_datatype.
inline void free_op(MPI_Op& op) noexcept { MPI_Op_free(&op); }

} // namespace mpi
} // namespace oti

#endif // OTINUM_MPI_HPP
