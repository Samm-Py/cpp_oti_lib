#pragma once

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
#include <cstring>
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

// --- value + location element for MAXLOC / MINLOC --------------------------
// A plain extremum value is insufficient for a jet: at an exact tie, equal real
// parts may carry different derivatives. Pair the jet with a unique application
// location (commonly an MPI rank or global mesh index), and resolve ties by the
// smaller location just like MPI_MAXLOC / MPI_MINLOC.
template <class T>
struct value_loc {
    T value{};
    int location = 0;
};

// Build and commit the MPI datatype for value_loc<T>. Unlike a bare otinum, this
// struct may contain alignment padding between/after its members, so describe
// the two fields explicitly and resize the datatype to the C++ array stride.
template <class T>
[[nodiscard]] MPI_Datatype make_value_loc_datatype()
{
    static_assert(is_tightly_packed_v<T>,
                  "value_loc requires a tightly packed otinum value");

    value_loc<T> sample{};
    MPI_Aint base = 0;
    MPI_Aint displacements[2] = {};
    MPI_Get_address(&sample, &base);
    MPI_Get_address(&sample.value[0], &displacements[0]);
    MPI_Get_address(&sample.location, &displacements[1]);
    displacements[0] -= base;
    displacements[1] -= base;

    const int block_lengths[2] = {T::ncoeffs, 1};
    MPI_Datatype types[2] = {
        mpi_scalar<typename T::coeff_type>::value(),
        MPI_INT,
    };

    MPI_Datatype fields;
    MPI_Type_create_struct(2, block_lengths, displacements, types, &fields);

    MPI_Datatype resized;
    MPI_Type_create_resized(fields, 0,
                            static_cast<MPI_Aint>(sizeof(value_loc<T>)),
                            &resized);
    MPI_Type_commit(&resized);
    MPI_Type_free(&fields);
    return resized;
}

// --- custom reduction operators --------------------------------------------
// MPI can MPI_SUM an int or a double, but it has no idea how to combine an
// otinum, so reductions (MPI_Reduce / MPI_Allreduce) over MPI_OTINUM need a user
// MPI_Op. make_reduce_op<T, Op>() turns any stateless, associative binary combine
// into one, so callers never hand-roll an MPI_User_function; the shipped
// combines below cover the common cases, and you pass your own Op for the rest.
//
// Op is a functor with `T operator()(const T& a, const T& b) const`. reduce_fn
// adapts it to MPI's type-erased, buffer-oriented callback: MPI hands two raw
// buffers of *len elements, and we apply Op across the batch. Op itself is the
// real jet arithmetic -- the loop is only the ABI glue.
//
// We must NOT reinterpret the MPI buffers as T* and index them directly. MPI
// builds the datatype from MPI_DOUBLE/MPI_INT, so it only guarantees those
// scalars' (<= 8-byte) alignment; the in/inout buffers it passes -- including
// internal reduction temporaries -- may be under-aligned for an over-aligned T
// (alignof(otinum) is 16 or, for 32-byte shapes, 32). A direct T& there is
// misaligned UB, and because the jet arithmetic emits aligned vector loads it
// can fault. So copy each element through a correctly-aligned local with memcpy
// (the standard idiom for a possibly-underaligned buffer); the copies usually
// optimize away, and the loads inside Op are then always well-aligned.
template <class T, class Op>
void reduce_fn(void* in, void* inout, int* len, MPI_Datatype* /*dt*/)
{
    const char* a = static_cast<const char*>(in);
    char* b = static_cast<char*>(inout);
    const int n = *len;
    const Op combine{};
    for (int i = 0; i < n; ++i) {
        T ai, bi;
        std::memcpy(&ai, a + static_cast<std::size_t>(i) * sizeof(T), sizeof(T));
        std::memcpy(&bi, b + static_cast<std::size_t>(i) * sizeof(T), sizeof(T));
        bi = combine(ai, bi);
        std::memcpy(b + static_cast<std::size_t>(i) * sizeof(T), &bi, sizeof(T));
    }
}

// Build AND commit an MPI_Op from a combine functor. The caller OWNS the returned
// handle and releases it with free_op (or MPI_Op_free).
template <class T, class Op>
[[nodiscard]] MPI_Op make_reduce_op(bool commute = true)
{
    MPI_Op op;
    MPI_Op_create(&reduce_fn<T, Op>, commute ? 1 : 0, &op);
    return op;
}

// Combines for the shipped operators. add/mul are otinum::operator+ / operator*:
// a sum is coefficient-wise, but a product is a convolution -- which is exactly
// why it cannot be faked with MPI_SUM over the raw scalars.
struct add_jets { template <class T> T operator()(T const& a, T const& b) const { return a + b; } };
struct mul_jets { template <class T> T operator()(T const& a, T const& b) const { return a * b; } };

// MAXLOC / MINLOC compare the real coefficient and carry the complete winning
// jet. Exact value ties select the smaller location, making the result explicit
// and deterministic when locations uniquely identify candidates.
struct maxloc_by_value {
    template <class Located>
    Located operator()(Located const& a, Located const& b) const
    {
        if (a.value[0] > b.value[0]) return a;
        if (b.value[0] > a.value[0]) return b;
        return a.location <= b.location ? a : b;
    }
};

struct minloc_by_value {
    template <class Located>
    Located operator()(Located const& a, Located const& b) const
    {
        if (a.value[0] < b.value[0]) return a;
        if (b.value[0] < a.value[0]) return b;
        return a.location <= b.location ? a : b;
    }
};

// Convenience builders for the common reductions. A reduction over MPI_OTINUM
// then yields a quantity of interest carrying its derivatives, with no hand-rolled
// user function. Extremum reductions operate on value_loc<T>, returning both the
// winning jet and its location with MPI_MAXLOC-style tie handling.
template <class T> [[nodiscard]] MPI_Op make_sum_op()  { return make_reduce_op<T, add_jets>(); }
template <class T> [[nodiscard]] MPI_Op make_prod_op() { return make_reduce_op<T, mul_jets>(); }
template <class T> [[nodiscard]] MPI_Op make_maxloc_op()
{
    return make_reduce_op<value_loc<T>, maxloc_by_value>();
}
template <class T> [[nodiscard]] MPI_Op make_minloc_op()
{
    return make_reduce_op<value_loc<T>, minloc_by_value>();
}

// Free an MPI_Op obtained from any make_*_op helper, symmetric with free_datatype.
inline void free_op(MPI_Op& op) noexcept { MPI_Op_free(&op); }

} // namespace mpi
} // namespace oti
