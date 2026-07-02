// Alignment-safety test for oti::mpi::reduce_fn (the MPI_Op user callback).
//
// MPI builds a reduction datatype from MPI_DOUBLE / MPI_INT and only guarantees
// those scalars' (<= 8 byte) alignment. The in/inout buffers it hands the user
// op -- including internal tree-reduction temporaries -- may therefore be
// under-aligned for an over-aligned otinum (alignof 16, or 32 for the byte-size-
// multiple-of-32 shapes promoted by the 32-byte alignment rung in core.hpp).
// reduce_fn must NOT bind those buffers as `otinum&` directly: that is misaligned
// UB and, because the jet arithmetic emits aligned vector loads, can fault. It
// copies each element through a correctly aligned local with memcpy instead.
//
// This test exercises exactly that: it places jet data at a deliberately
// UNDER-aligned offset (8-aligned, not 32) and runs reduce_fn over it, for the
// promoted alignas(32) shapes and an aligned control. It verifies the values are
// correct AND -- crucially -- must be built with -fsanitize=undefined so the
// alignment checker aborts if reduce_fn ever regresses to a direct otinum& bind.
// (No MPI runtime is needed; this is a single-process unit test.)
//
// Build: mpicxx -std=c++17 -O2 -fsanitize=undefined -fno-sanitize-recover=undefined \
//               -I ../../../include test_reduce_alignment.cpp -o test_reduce_alignment
// Run:   ./test_reduce_alignment

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

// Place n jets at a deliberately under-aligned offset inside buf: 32-byte align
// the block, then push by 8 -- valid for the underlying doubles/floats, but NOT
// 32-aligned, so a direct otinum& bind would be misaligned for alignas(32) T.
template <class T>
static T* misaligned_fill(std::vector<char>& buf, int n, double base)
{
    const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(buf.data());
    const std::uintptr_t aligned = (p + 31) & ~std::uintptr_t(31);
    char* start = reinterpret_cast<char*>(aligned + 8);
    for (int i = 0; i < n; ++i) {
        T v{};
        for (int k = 0; k < T::ncoeffs; ++k) v[k] = base + i + 0.5 * k;
        std::memcpy(start + static_cast<std::size_t>(i) * sizeof(T), &v, sizeof(T));
    }
    return reinterpret_cast<T*>(start);
}

template <class T>
static bool sum_on_misaligned(const char* tag, int n)
{
    std::vector<char> bin(static_cast<std::size_t>(n) * sizeof(T) + 64);
    std::vector<char> bio(static_cast<std::size_t>(n) * sizeof(T) + 64);
    T* in    = misaligned_fill<T>(bin, n, 1.0);
    T* inout = misaligned_fill<T>(bio, n, 100.0);

    MPI_Datatype dt = MPI_DOUBLE;  // unused by reduce_fn; just a valid handle
    int len = n;
    oti::mpi::reduce_fn<T, oti::mpi::add_jets>(in, inout, &len, &dt);

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        T got{};
        std::memcpy(&got, reinterpret_cast<char*>(inout) + static_cast<std::size_t>(i) * sizeof(T),
                    sizeof(T));
        for (int k = 0; k < T::ncoeffs; ++k) {
            const double exp = (100.0 + i + 0.5 * k) + (1.0 + i + 0.5 * k);
            if (got[k] != exp) ok = false;
        }
    }
    std::printf("  %-10s n=%d  alignof=%zu  %s\n", tag, n, alignof(T), ok ? "ok" : "VALUE MISMATCH");
    return ok;
}

int main()
{
    std::printf("reduce_fn on under-aligned buffers (build with -fsanitize=undefined):\n");
    bool ok = true;
    ok &= sum_on_misaligned<oti::otinum<3, 1, double>>("<3,1>d*", 7);  // promoted alignas(32)
    ok &= sum_on_misaligned<oti::otinum<3, 3, double>>("<3,3>d*", 5);  // promoted alignas(32)
    ok &= sum_on_misaligned<oti::otinum<7, 1, float>>("<7,1>f*", 9);   // promoted alignas(32)
    ok &= sum_on_misaligned<oti::otinum<2, 2, double>>("<2,2>d", 8);   // control alignas(16)
    std::printf("%s\n", ok ? "reduce alignment test passed" : "reduce alignment test FAILED");
    return ok ? 0 : 1;
}
