#include "gol_engine.h"

#include <cstdint>
#include <cstring>
#include <omp.h>

#if __cpp_lib_simd
#include <simd>
#define GOL_USE_SIMD_LIB 1
namespace {
using std::simd_flag_default;
using std::simd_select;
} // namespace
#elif __cplusplus > 202302L && defined(__GNUC__) && !defined(__clang__) &&     \
    __has_include(<experimental/simd>)
#include <experimental/simd>
#define GOL_USE_SIMD_LIB 1
namespace std {
using namespace std::experimental;
}
namespace {
constexpr auto simd_flag_default = std::experimental::element_aligned;
template <class M, class V> V simd_select(M mask, V a, V b) {
    V r                               = b;
    std::experimental::where(mask, r) = a;
    return r;
}
} // namespace
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX512BW__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// ── SIMD helpers ─────────────────────────────────────────────────────────────

uint8_t SIMDGameOfLife::golRule(uint8_t alive, uint8_t nb) {
    return (nb == 3) | (alive & (nb == 2));
}

template <bool HasPrev, bool HasNext, bool HasLeft, bool HasRight>
uint8_t SIMDGameOfLife::aliveNeighbours(const uint8_t *p, const uint8_t *c,
                                        const uint8_t *n, size_t col) {
    uint8_t nb = 0;
    if constexpr (HasPrev) {
        if constexpr (HasLeft)
            nb += p[col - 1];
        nb += p[col];
        if constexpr (HasRight)
            nb += p[col + 1];
    }
    if constexpr (HasLeft)
        nb += c[col - 1];
    if constexpr (HasRight)
        nb += c[col + 1];
    if constexpr (HasNext) {
        if constexpr (HasLeft)
            nb += n[col - 1];
        nb += n[col];
        if constexpr (HasRight)
            nb += n[col + 1];
    }
    return nb;
}

// Explicit instantiations for all 16 combinations of boundary booleans.
template uint8_t SIMDGameOfLife::aliveNeighbours<false, false, false, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, false, false, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, false, true, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, false, true, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, true, false, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, true, false, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, true, true, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<false, true, true, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, false, false, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, false, false, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, false, true, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, false, true, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, true, false, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, true, false, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, true, true, false>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);
template uint8_t SIMDGameOfLife::aliveNeighbours<true, true, true, true>(
    const uint8_t *, const uint8_t *, const uint8_t *, size_t);

// Processes interior columns [1, cols-1) for a fully-surrounded interior row.
void SIMDGameOfLife::processInteriorCells(const uint8_t *__restrict__ p,
                                          const uint8_t *__restrict__ c,
                                          const uint8_t *__restrict__ n,
                                          uint8_t *__restrict__ o,
                                          size_t cols) {
#if GOL_USE_SIMD_LIB
    // ── std::simd (C++26) / std::experimental::simd ────────────────────────
    using Vec          = std::native_simd<uint8_t>;
    constexpr size_t W = Vec::size();
    size_t col         = 1;
    for (; col + W <= cols - 1; col += W) {
        __builtin_prefetch(p + col + W * 4, 0, 0);
        __builtin_prefetch(c + col + W * 4, 0, 0);
        __builtin_prefetch(n + col + W * 4, 0, 0);
        const Vec pL(p + col - 1, simd_flag_default);
        const Vec pC(p + col, simd_flag_default);
        const Vec pR(p + col + 1, simd_flag_default);
        const Vec cL(c + col - 1, simd_flag_default);
        const Vec cC(c + col, simd_flag_default);
        const Vec cR(c + col + 1, simd_flag_default);
        const Vec nL(n + col - 1, simd_flag_default);
        const Vec nC(n + col, simd_flag_default);
        const Vec nR(n + col + 1, simd_flag_default);

        const Vec nb = pL + pC + pR + cL + cR + nL + nC + nR;

        const Vec result =
            simd_select((nb == Vec(uint8_t(3))) |
                            ((cC != Vec(uint8_t(0))) & (nb == Vec(uint8_t(2)))),
                        Vec(uint8_t(1)), Vec(uint8_t(0)));

        result.copy_to(o + col, simd_flag_default);
    }
    for (; col < cols - 1; ++col)
        o[col] = golRule(c[col],
                         aliveNeighbours<true, true, true, true>(p, c, n, col));

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
    // ── ARM NEON intrinsics ───────────────────────────────────────────────
    constexpr size_t W = 16;
    size_t col         = 1;
    for (; col + W <= cols - 1; col += W) {
        __builtin_prefetch(p + col + W * 4, 0, 0);
        __builtin_prefetch(c + col + W * 4, 0, 0);
        __builtin_prefetch(n + col + W * 4, 0, 0);
        const uint8x16_t pL = vld1q_u8(p + col - 1);
        const uint8x16_t pC = vld1q_u8(p + col);
        const uint8x16_t pR = vld1q_u8(p + col + 1);
        const uint8x16_t cL = vld1q_u8(c + col - 1);
        const uint8x16_t cC = vld1q_u8(c + col);
        const uint8x16_t cR = vld1q_u8(c + col + 1);
        const uint8x16_t nL = vld1q_u8(n + col - 1);
        const uint8x16_t nC = vld1q_u8(n + col);
        const uint8x16_t nR = vld1q_u8(n + col + 1);

        uint8x16_t nb = vaddq_u8(pL, pC);
        nb            = vaddq_u8(nb, pR);
        nb            = vaddq_u8(nb, cL);
        nb            = vaddq_u8(nb, cR);
        nb            = vaddq_u8(nb, nL);
        nb            = vaddq_u8(nb, nC);
        nb            = vaddq_u8(nb, nR);

        const uint8x16_t born = vceqq_u8(nb, vdupq_n_u8(3));
        const uint8x16_t survive =
            vandq_u8(vtstq_u8(cC, cC), vceqq_u8(nb, vdupq_n_u8(2)));
        const uint8x16_t result =
            vandq_u8(vorrq_u8(born, survive), vdupq_n_u8(1));

        vst1q_u8(o + col, result);
    }
    for (; col < cols - 1; ++col)
        o[col] = golRule(c[col],
                         aliveNeighbours<true, true, true, true>(p, c, n, col));

#elif defined(__AVX2__)
    // ── AVX2 intrinsics ───────────────────────────────────────────────────
    constexpr size_t W = 32;
    size_t col         = 1;
    for (; col + W <= cols - 1; col += W) {
        __builtin_prefetch(p + col + W * 4, 0, 0);
        __builtin_prefetch(c + col + W * 4, 0, 0);
        __builtin_prefetch(n + col + W * 4, 0, 0);
        const __m256i pL = _mm256_loadu_si256((const __m256i *)(p + col - 1));
        const __m256i pC = _mm256_loadu_si256((const __m256i *)(p + col));
        const __m256i pR = _mm256_loadu_si256((const __m256i *)(p + col + 1));
        const __m256i cL = _mm256_loadu_si256((const __m256i *)(c + col - 1));
        const __m256i cC = _mm256_loadu_si256((const __m256i *)(c + col));
        const __m256i cR = _mm256_loadu_si256((const __m256i *)(c + col + 1));
        const __m256i nL = _mm256_loadu_si256((const __m256i *)(n + col - 1));
        const __m256i nC = _mm256_loadu_si256((const __m256i *)(n + col));
        const __m256i nR = _mm256_loadu_si256((const __m256i *)(n + col + 1));

        __m256i nb = _mm256_add_epi8(pL, pC);
        nb         = _mm256_add_epi8(nb, pR);
        nb         = _mm256_add_epi8(nb, cL);
        nb         = _mm256_add_epi8(nb, cR);
        nb         = _mm256_add_epi8(nb, nL);
        nb         = _mm256_add_epi8(nb, nC);
        nb         = _mm256_add_epi8(nb, nR);

        const __m256i born = _mm256_cmpeq_epi8(nb, _mm256_set1_epi8(3));
        const __m256i survive =
            _mm256_and_si256(_mm256_cmpgt_epi8(cC, _mm256_setzero_si256()),
                             _mm256_cmpeq_epi8(nb, _mm256_set1_epi8(2)));
        const __m256i result = _mm256_and_si256(_mm256_or_si256(born, survive),
                                                _mm256_set1_epi8(1));

        _mm256_storeu_si256((__m256i *)(o + col), result);
    }
    for (; col < cols - 1; ++col)
        o[col] = golRule(c[col],
                         aliveNeighbours<true, true, true, true>(p, c, n, col));

#elif defined(__AVX512BW__)
    // ── AVX512BW intrinsics ───────────────────────────────────────────────
    constexpr size_t W = 64;
    size_t col         = 1;
    for (; col + W <= cols - 1; col += W) {
        __builtin_prefetch(p + col + W * 4, 0, 0);
        __builtin_prefetch(c + col + W * 4, 0, 0);
        __builtin_prefetch(n + col + W * 4, 0, 0);
        const __m512i pL = _mm512_loadu_si512(p + col - 1);
        const __m512i pC = _mm512_loadu_si512(p + col);
        const __m512i pR = _mm512_loadu_si512(p + col + 1);
        const __m512i cL = _mm512_loadu_si512(c + col - 1);
        const __m512i cC = _mm512_loadu_si512(c + col);
        const __m512i cR = _mm512_loadu_si512(c + col + 1);
        const __m512i nL = _mm512_loadu_si512(n + col - 1);
        const __m512i nC = _mm512_loadu_si512(n + col);
        const __m512i nR = _mm512_loadu_si512(n + col + 1);

        __m512i nb = _mm512_add_epi8(pL, pC);
        nb         = _mm512_add_epi8(nb, pR);
        nb         = _mm512_add_epi8(nb, cL);
        nb         = _mm512_add_epi8(nb, cR);
        nb         = _mm512_add_epi8(nb, nL);
        nb         = _mm512_add_epi8(nb, nC);
        nb         = _mm512_add_epi8(nb, nR);

        const __mmask64 born = _mm512_cmpeq_epi8_mask(nb, _mm512_set1_epi8(3));
        const __mmask64 survive =
            _kand_mask64(_mm512_cmpgt_epi8_mask(cC, _mm512_setzero_si512()),
                         _mm512_cmpeq_epi8_mask(nb, _mm512_set1_epi8(2)));
        const __m512i result =
            _mm512_maskz_set1_epi8(_kor_mask64(born, survive), 1);

        _mm512_storeu_si512(o + col, result);
    }
    for (; col < cols - 1; ++col)
        o[col] = golRule(c[col],
                         aliveNeighbours<true, true, true, true>(p, c, n, col));

#else
    // ── Scalar fallback ───────────────────────────────────────────────────
    for (size_t col = 1; col < cols - 1; ++col)
        o[col] = golRule(c[col],
                         aliveNeighbours<true, true, true, true>(p, c, n, col));
#endif
}

void SIMDGameOfLife::processInteriorRow(const uint8_t *p, const uint8_t *c,
                                        const uint8_t *n, uint8_t *o,
                                        size_t cols) {
    o[0] = golRule(c[0], aliveNeighbours<true, true, false, true>(p, c, n, 0));
    o[cols - 1] =
        golRule(c[cols - 1],
                aliveNeighbours<true, true, true, false>(p, c, n, cols - 1));
    processInteriorCells(p, c, n, o, cols);
}

template <bool HasPrev, bool HasNext>
void SIMDGameOfLife::processBorderRow(const uint8_t *p, const uint8_t *c,
                                      const uint8_t *n, uint8_t *o,
                                      size_t cols) {
    o[0] = golRule(c[0],
                   aliveNeighbours<HasPrev, HasNext, false, true>(p, c, n, 0));
    for (size_t col = 1; col + 1 < cols; ++col)
        o[col] = golRule(c[col], aliveNeighbours<HasPrev, HasNext, true, true>(
                                     p, c, n, col));
    o[cols - 1] = golRule(
        c[cols - 1],
        aliveNeighbours<HasPrev, HasNext, true, false>(p, c, n, cols - 1));
}

template void SIMDGameOfLife::processBorderRow<false, true>(const uint8_t *,
                                                            const uint8_t *,
                                                            const uint8_t *,
                                                            uint8_t *, size_t);
template void SIMDGameOfLife::processBorderRow<true, false>(const uint8_t *,
                                                            const uint8_t *,
                                                            const uint8_t *,
                                                            uint8_t *, size_t);

// ── SIMDGameOfLife implementation ────────────────────────────────────────────

SIMDGameOfLife::SIMDGameOfLife(Grid &grid)
    : currentGrid_(std::move(grid)),
      newGrid_(currentGrid_.getNumRows(), currentGrid_.getNumCols()) {
    rows_ = currentGrid_.getNumRows();
    cols_ = currentGrid_.getNumCols();
}

void SIMDGameOfLife::takeStep() {
    const uint8_t *__restrict__ src = currentGrid_.getData();
    uint8_t *__restrict__ dst       = newGrid_.getData();
    const size_t rows               = rows_;
    const size_t cols               = cols_;

#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t *p = (row > 0) ? src + (row - 1) * cols : nullptr;
        const uint8_t *c = src + row * cols;
        const uint8_t *n = (row < rows - 1) ? src + (row + 1) * cols : nullptr;
        uint8_t *o       = dst + row * cols;

        if (row == 0)
            processBorderRow<false, true>(p, c, n, o, cols);
        else if (row == rows - 1)
            processBorderRow<true, false>(p, c, n, o, cols);
        else
            processInteriorRow(p, c, n, o, cols);
    }
    currentGrid_.swap(newGrid_);
}

void *SIMDGameOfLife::getRowDataRaw(size_t row) {
    return currentGrid_.getRowData(row);
}

size_t SIMDGameOfLife::getStride() const {
    return cols_;
}

CellKind SIMDGameOfLife::getCellKind() const {
    return CellKind::Byte;
}
