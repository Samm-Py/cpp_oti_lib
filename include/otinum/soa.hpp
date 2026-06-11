#pragma once

// Coefficient-major (structure-of-arrays) storage view for otinum arrays.
//
// An array of otinum<M, N, Coeff> normally stores each number's coefficients
// contiguously (array-of-structs): element i occupies one ncoeffs-sized block.
// This header provides oti::soa_span, a non-owning view over the transposed
// layout: all elements' coefficient k are contiguous, so coefficient k of
// element i lives at data[k * extent + i].
//
// The transposition matters for throughput, not semantics. When many threads
// (a GPU warp, a SIMD loop) each process one element and walk its coefficients
// in lockstep, the AoS layout makes "everyone reads coefficient k" a strided
// access spanning one cache line per thread, while the SoA layout makes it a
// single coalesced transaction over adjacent addresses.
//
// Which layout is faster is backend-dependent (measured with
// tests/benchmark_soa_layout.cpp, streaming ~96 MiB arrays):
//
// - GPU (CUDA): SoA never loses, and wins large exactly where the AoS strided
//   pattern falls off the bandwidth peak -- on a GTX 1650, <3,3> double mul
//   2.9x and <4,4> float mul 4.2x, with parity (1.0x) for every <3,1> shape
//   (AoS is already coalesced enough there thanks to the aligned 2x16-byte
//   jets). Which AoS shapes collapse is hard to predict (<3,3> double does,
//   <3,3> float does not), which is itself a reason to prefer SoA on GPU for
//   N >= 2: it sits at the bandwidth peak predictably.
//
// - CPU (OpenMP): SoA LOSES everywhere (0.5-0.85x). Each thread walks
//   elements sequentially, so AoS is one perfectly prefetchable stream per
//   thread, while the SoA gather splits it into ncoeffs strided streams that
//   defeat the hardware prefetcher. Keep AoS Views on CPU.
//
// The view does not change the value type: kernels load() an element into a
// register-resident otinum, run the ordinary unrolled arithmetic, and store()
// the result back. Because the arithmetic is identical, results are
// bit-identical to the AoS form; only the memory access pattern differs.
//
// The span is non-owning and trivially copyable, so it can be captured by
// value in device lambdas. The caller allocates the flat coefficient buffer
// (e.g. a Kokkos::View<Coeff*> of size soa_span::required_size(n)) and keeps
// it alive for the span's lifetime.

#include <cstddef>

#include "otinum/core.hpp"

namespace oti {

template <int M, int N, class Coeff = double>
class soa_span {
public:
    using value_type = otinum<M, N, Coeff>;
    using coeff_type = Coeff;

    static constexpr int ncoeffs = value_type::ncoeffs;

    OTI_CONSTEXPR_FUNCTION soa_span() noexcept : data_(nullptr), extent_(0) {}

    // View over a caller-owned buffer of at least required_size(extent)
    // coefficients holding `extent` otinums in coefficient-major order.
    OTI_CONSTEXPR_FUNCTION soa_span(Coeff* data, std::size_t extent) noexcept
        : data_(data),
          extent_(extent)
    {
    }

    // Number of coefficients a backing buffer must hold for `extent` elements.
    static OTI_CONSTEXPR_FUNCTION std::size_t required_size(std::size_t extent) noexcept
    {
        return extent * static_cast<std::size_t>(ncoeffs);
    }

    OTI_CONSTEXPR_FUNCTION std::size_t extent() const noexcept
    {
        return extent_;
    }

    OTI_CONSTEXPR_FUNCTION Coeff* data() const noexcept
    {
        return data_;
    }

    // Gather element i into a register-resident otinum.
    OTI_CONSTEXPR_FUNCTION value_type load(std::size_t i) const noexcept
    {
        OTI_ASSERT(i < extent_);
        value_type out;
        for (int k = 0; k < ncoeffs; ++k) {
            out[k] = data_[static_cast<std::size_t>(k) * extent_ + i];
        }
        return out;
    }

    // Scatter a register-resident otinum back to element i.
    OTI_CONSTEXPR_FUNCTION void store(std::size_t i, value_type const& value) const noexcept
    {
        OTI_ASSERT(i < extent_);
        for (int k = 0; k < ncoeffs; ++k) {
            data_[static_cast<std::size_t>(k) * extent_ + i] = value[k];
        }
    }

    // Direct access to one coefficient of one element, without materializing
    // the whole jet. Useful for initialization and coefficient-wise kernels
    // (which see the layout's full coalescing benefit with no gather at all).
    OTI_CONSTEXPR_FUNCTION Coeff& coeff(std::size_t i, int k) const noexcept
    {
        OTI_ASSERT(i < extent_ && k >= 0 && k < ncoeffs);
        return data_[static_cast<std::size_t>(k) * extent_ + i];
    }

private:
    Coeff* data_;
    std::size_t extent_;
};

} // namespace oti
