#include "gol_gpu.h"
#include "gol_gpu_test_wrappers.h"

#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    hipGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static void hipAlloc(void **ptr, size_t bytes) {
    HIP_CHECK(hipMalloc(ptr, bytes));
}
static void hipFreeWrap(void *ptr) {
    (void)hipFree(ptr);
}
static void hipCopyH2D(void *dst, const void *src, size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}
static void hipCopyD2H(void *dst, const void *src, size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}
static void hipCopy2D_H2D(void *dst, size_t dpitch, const void *src,
                          size_t spitch, size_t width, size_t height) {
    HIP_CHECK(hipMemcpy2D(dst, dpitch, src, spitch, width, height,
                          hipMemcpyHostToDevice));
}
static void hipCopy2D_D2H(void *dst, size_t dpitch, const void *src,
                          size_t spitch, size_t width, size_t height) {
    HIP_CHECK(hipMemcpy2D(dst, dpitch, src, spitch, width, height,
                          hipMemcpyDeviceToHost));
}
static void hipMemsetWrap(void *ptr, int value, size_t bytes) {
    HIP_CHECK(hipMemset(ptr, value, bytes));
}
static void hipSyncWrap() {
    HIP_CHECK(hipDeviceSynchronize());
}
static void hipCheckLastWrap(const char *ctx) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error after %s: %s\n", ctx,
                hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static GpuOps hipOps() {
    return {hipAlloc,      hipFreeWrap,   hipCopyH2D,
            hipCopyD2H,    hipCopy2D_H2D, hipCopy2D_D2H,
            hipMemsetWrap, hipSyncWrap,   hipCheckLastWrap};
}

// ===========================================================================
// Byte-per-cell kernels — two engines
// ===========================================================================
//
//   1. hipGolKernelSimple — one thread per cell, reads neighbours directly
//                           from global memory.  No shared memory.
//   2. hipGolKernelTile   — shared memory tiling (256 × 8 tiles) with uint32_t
//                           vectorised loads and stores.
//
// Both kernels use the same branch-free GoL rule.
//
// All byte kernels take separate `cols` and `stride` parameters.
// `cols` is the logical grid width (used for boundary checks).
// `stride` is the device buffer row pitch in bytes (>= cols, always a
// multiple of 4 so that uint32_t loads/stores are naturally aligned).
// The padding columns between cols and stride are kept at zero (dead cells)
// and are never written by the kernels.

// Branch-free GoL rule: count 8 neighbours, apply birth/survival.
// Used directly by the tiled kernels (from shared memory tile), and as the
// logic reference for the simple kernel (which reads from global memory).
__device__ uint8_t golRule(const uint8_t *tile, unsigned int tileW,
                           unsigned int tx, unsigned int ty) {
    unsigned int nb =
        tile[(ty - 1) * tileW + tx - 1] + tile[(ty - 1) * tileW + tx] +
        tile[(ty - 1) * tileW + tx + 1] + tile[ty * tileW + tx - 1] +
        tile[ty * tileW + tx + 1] + tile[(ty + 1) * tileW + tx - 1] +
        tile[(ty + 1) * tileW + tx] + tile[(ty + 1) * tileW + tx + 1];
    unsigned int alive = tile[ty * tileW + tx];
    return (nb == 3) | (alive & (nb == 2));
}

// -- 1. Simple kernel (hip-simple) ----------------------------------------
//
// The most direct GPU translation of the CPU approach.
//
// On the CPU, `#pragma omp parallel for` dispatches rows to threads, and
// each thread walks columns left-to-right.  Here we go one step further:
// one thread per cell, no loops at all.  The kernel grid covers the entire
// domain — each thread computes exactly one output cell by reading its 8
// neighbours from global memory.
//
// Block: 64 × 4 = 256 threads.  The 64-thread wavefront dimension is along
// columns, so adjacent threads access adjacent bytes — coalesced reads
// and writes.
//
// Interior / border split:  On the CPU, boundary checks are eliminated at
// compile time via templates (HasPrev, HasNext, etc.) — the compiler
// generates separate code paths for border and interior rows.  We do the
// same on the GPU by launching two separate kernels:
//
//   1. Interior kernel — covers cells at row [1, rows-2], col [1, cols-2].
//      The launch grid is rounded down to full blocks, so every thread
//      maps to a valid interior cell.  Zero conditionals: 8 loads +
//      golRule + 1 store.  Remaining interior cells in partial blocks
//      at the right/bottom edges are handled by the border kernel.
//
//   2. Border kernel — 1D launch over the perimeter cells (top/bottom rows,
//      left/right columns) plus any interior cells in partial blocks not
//      covered by the full-block interior launch.  Has boundary checks, but
//      handles < 0.01% of cells for any reasonably sized grid.
//
// This split pays off here because the simple kernel evaluates boundary
// conditionals per-cell (8 branches × N cells).  The tiled kernels (§2–§3)
// don't need it — their boundary checks are per-tile (once per block), so
// the cost of a second kernel launch outweighs the savings.
//
// Each cell in the grid may be read from global memory up to 9 times
// (once as itself, up to 8 times as a neighbour of adjacent cells).
// On a CPU, the hardware cache absorbs this reuse transparently —
// three active rows fit in L2, and the prefetcher keeps them warm.
// On a GPU, the per-CU L1 cache is small and shared across many concurrent
// wavefronts on the same CU, so cache lines are often evicted before they
// can be reused.
//
// The tiled kernel (§2) fixes this by loading each cell exactly once into
// shared memory (LDS) — a programmer-managed on-chip store that is private
// to each block and not subject to eviction by other blocks' traffic.

static constexpr int HIP_SIMPLE_BLOCK_X = 64;
static constexpr int HIP_SIMPLE_BLOCK_Y = 4;

// Interior kernel — zero conditionals.  Every thread is guaranteed to map
// to a valid interior cell (row in [1, rows-2], col in [1, cols-2]).
// The launch grid is rounded down to full blocks only.
__global__ void hipGolKernelSimpleInterior(const uint8_t *src, uint8_t *dst,
                                           unsigned int cols,
                                           unsigned int stride) {
    unsigned int col = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = row * stride + col;

    unsigned int up = idx - stride;
    unsigned int dn = idx + stride;
    unsigned int nb = src[up - 1] + src[up] + src[up + 1] + src[idx - 1] +
                      src[idx + 1] + src[dn - 1] + src[dn] + src[dn + 1];

    unsigned int alive = src[idx];
    dst[idx]           = (nb == 3) | (alive & (nb == 2));
}

// Border kernel — handles the 4 perimeter strips plus any interior cells
// not covered by the full-block interior launch (partial-block remainders
// at the right and bottom edges).  Cell count is negligible so the branches
// have no measurable impact.
__global__ void hipGolKernelSimpleBorder(const uint8_t *src, uint8_t *dst,
                                         unsigned int rows, unsigned int cols,
                                         unsigned int stride,
                                         unsigned int interiorCols,
                                         unsigned int interiorRows) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perimeter: top row (cols) + bottom row (cols) + left column (rows-2) +
    //            right column (rows-2) = 2*cols + 2*(rows-2)
    unsigned int perimeterCells = 2 * cols + 2 * (rows - 2);

    // Remainder strips: interior cells not covered by the full-block launch.
    // Right strip:  rows in [1, rows-2], cols in [1+interiorCols, cols-2]
    // Bottom strip: rows in [1+interiorRows, rows-2], cols in [1,
    // 1+interiorCols-1]
    unsigned int rightStripW =
        (cols >= 2 + interiorCols) ? (cols - 2 - interiorCols) : 0;
    unsigned int rightStripCells = (rows >= 2) ? rightStripW * (rows - 2) : 0;
    unsigned int bottomStripH =
        (rows >= 2 + interiorRows) ? (rows - 2 - interiorRows) : 0;
    unsigned int bottomStripCells = bottomStripH * interiorCols;

    unsigned int totalCells =
        perimeterCells + rightStripCells + bottomStripCells;
    if (tid >= totalCells)
        return;

    unsigned int row, col;
    if (tid < perimeterCells) {
        // Map tid to a perimeter cell
        if (tid < cols) {
            // Top row
            row = 0;
            col = tid;
        } else if (tid < 2 * cols) {
            // Bottom row
            row = rows - 1;
            col = tid - cols;
        } else {
            unsigned int side = tid - 2 * cols;
            if (side < rows - 2) {
                // Left column (excluding corners)
                row = 1 + side;
                col = 0;
            } else {
                // Right column (excluding corners)
                row = 1 + (side - (rows - 2));
                col = cols - 1;
            }
        }
    } else if (tid < perimeterCells + rightStripCells) {
        // Right remainder strip
        unsigned int s = tid - perimeterCells;
        row            = 1 + s / rightStripW;
        col            = 1 + interiorCols + s % rightStripW;
    } else {
        // Bottom remainder strip
        unsigned int s = tid - perimeterCells - rightStripCells;
        row            = 1 + interiorRows + s / interiorCols;
        col            = 1 + s % interiorCols;
    }

    unsigned int idx = row * stride + col;
    unsigned int nb  = 0;
    bool hasUp       = row > 0;
    bool hasDown     = row < rows - 1;
    bool hasLeft     = col > 0;
    bool hasRight    = col < cols - 1;

    if (hasUp) {
        unsigned int up = idx - stride;
        if (hasLeft)
            nb += src[up - 1];
        nb += src[up];
        if (hasRight)
            nb += src[up + 1];
    }
    if (hasLeft)
        nb += src[idx - 1];
    if (hasRight)
        nb += src[idx + 1];
    if (hasDown) {
        unsigned int dn = idx + stride;
        if (hasLeft)
            nb += src[dn - 1];
        nb += src[dn];
        if (hasRight)
            nb += src[dn + 1];
    }

    unsigned int alive = src[idx];
    dst[idx]           = (nb == 3) | (alive & (nb == 2));
}

HIPSimpleGameOfLife::HIPSimpleGameOfLife(Grid &grid)
    : GPUEngine(std::move(grid), hipOps()) {
}

CellKind HIPSimpleGameOfLife::getCellKind() const {
    return CellKind::Byte;
}

void HIPSimpleGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
    unsigned int r = static_cast<unsigned int>(rows_);
    unsigned int c = static_cast<unsigned int>(cols_);
    unsigned int s = static_cast<unsigned int>(deviceStride_);

    // Interior: only full blocks — every thread maps to a valid cell.
    unsigned int intCols = ((c - 2) / HIP_SIMPLE_BLOCK_X) * HIP_SIMPLE_BLOCK_X;
    unsigned int intRows = ((r - 2) / HIP_SIMPLE_BLOCK_Y) * HIP_SIMPLE_BLOCK_Y;

    if (intCols > 0 && intRows > 0) {
        dim3 block(HIP_SIMPLE_BLOCK_X, HIP_SIMPLE_BLOCK_Y);
        dim3 grid(intCols / HIP_SIMPLE_BLOCK_X, intRows / HIP_SIMPLE_BLOCK_Y);
        hipGolKernelSimpleInterior<<<grid, block>>>(src, dst, c, s);
        HIP_CHECK(hipGetLastError());
    }

    // Border + remainder strips: 1D launch over all remaining cells.
    unsigned int perim       = 2 * c + 2 * (r - 2);
    unsigned int rightW      = (c >= 2 + intCols) ? (c - 2 - intCols) : 0;
    unsigned int rightCells  = (r >= 2) ? rightW * (r - 2) : 0;
    unsigned int bottomH     = (r >= 2 + intRows) ? (r - 2 - intRows) : 0;
    unsigned int bottomCells = bottomH * intCols;
    unsigned int totalBorder = perim + rightCells + bottomCells;

    if (totalBorder > 0) {
        int bThreads = 256;
        int bBlocks  = (totalBorder + bThreads - 1) / bThreads;
        hipGolKernelSimpleBorder<<<bBlocks, bThreads>>>(src, dst, r, c, s,
                                                        intCols, intRows);
        HIP_CHECK(hipGetLastError());
    }
}

// -- 2. Tiled kernel (hip-tile) -------------------------------------------
//
// The simple kernel reads each cell from global memory up to 9 times (as a
// neighbour of surrounding cells). Shared memory (LDS) eliminates this
// redundancy: each block cooperatively loads its tile (including a 1-cell
// halo) from global memory into on-chip LDS, then all neighbour reads come
// from LDS at ~100× lower latency.
//
// Unlike the per-CU L1 cache, LDS is explicitly managed and private to each
// block — data stays resident until the block is done with it, regardless of
// what other blocks on the same CU are doing.
//
// Block: 64 × 8 = 512 threads. Each thread computes 4 consecutive cells in
// the column direction, so a block covers 256 columns × 8 rows. Tile:
// (256 + 2) × (8 + 2) = 2,580 bytes shared memory (incl. halo). Grid: 2D
// launch, one block per tile — all tiles run concurrently.
//
// Loads: uint32_t vectorised reads for the 256 interior tile columns (4 bytes
// at a time), with scalar reads for the 2 halo columns. 256 cols / 4 = 64
// words per row × 10 rows = 640 words total. Division by 64 (power of 2)
// compiles to a shift. Within a wavefront, 64 threads each load one uint32_t
// from consecutive addresses → 256 contiguous bytes, perfectly coalesced.
//
// Stores: each thread packs its 4 results into a uint32_t and writes it with
// a single 4-byte store. Within a wavefront, 64 threads × 4 bytes = 256
// contiguous bytes, perfectly coalesced. Byte stores cannot merge into a
// single wide transaction the way a uint32_t store does, so the packing is
// critical for write throughput.
//
// Boundary handling: the last tile column (when cols is not a multiple of
// 256) falls through to byte-by-byte load and store paths. This affects only
// 1 out of ceil(cols/256) tile columns, so the branch cost is negligible.
//
// Stride padding: the device buffer row pitch (stride) is rounded up to a
// multiple of 4 bytes so that every row starts at a uint32_t-aligned address.
// Padding columns are kept at zero and are never written by the kernel.
//
static constexpr int HIP_BYTE_BLOCK_X          = 64;
static constexpr int HIP_BYTE_BLOCK_Y          = 8;
static constexpr int HIP_BYTE_CELLS_PER_THREAD = 4;
static constexpr int HIP_BYTE_TILE_COLS =
    HIP_BYTE_BLOCK_X * HIP_BYTE_CELLS_PER_THREAD; // 256

__global__ void hipGolKernelTile(const uint8_t *src, uint8_t *dst,
                                 unsigned int rows, unsigned int cols,
                                 unsigned int stride) {
    extern __shared__ uint8_t tile[];
    const unsigned int tileW     = HIP_BYTE_TILE_COLS + 2; // 258
    const unsigned int tileH     = blockDim.y + 2;         // 10
    const unsigned int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int blockSize = blockDim.x * blockDim.y; // 512

    // Interior columns start at global column gc0_interior.
    // Halo left column is gc0_interior - 1, halo right is gc0_interior + 256.
    const unsigned int gc0_interior = blockIdx.x * HIP_BYTE_TILE_COLS;
    const int gr0                   = (int)(blockIdx.y * blockDim.y) - 1;

    // Interior load (uint32_t vectorised) --
    const unsigned int wordsPerRow = HIP_BYTE_TILE_COLS / 4; // 64
    const unsigned int totalWords  = wordsPerRow * tileH;    // 640
    for (unsigned int i = tid; i < totalWords; i += blockSize) {
        unsigned int tileRow = i >> 6; // i / 64
        unsigned int wordIdx = i & 63; // i % 64
        int gr               = gr0 + (int)tileRow;
        unsigned int gc      = gc0_interior + wordIdx * 4;
        uint32_t packed      = 0;
        if (gr >= 0 && (unsigned)gr < rows && gc + 3 < cols)
            packed = *reinterpret_cast<const uint32_t *>(
                &src[(unsigned)gr * stride + gc]);
        else if (gr >= 0 && (unsigned)gr < rows && gc < cols) {
            // Partial word at right edge — load available bytes
            for (unsigned int b = 0; b < 4 && gc + b < cols; b++)
                packed |= (uint32_t)src[(unsigned)gr * stride + gc + b]
                          << (b * 8);
        }
        unsigned int tileBase = tileRow * tileW + 1 + wordIdx * 4;
        tile[tileBase + 0]    = packed & 0xFF;
        tile[tileBase + 1]    = (packed >> 8) & 0xFF;
        tile[tileBase + 2]    = (packed >> 16) & 0xFF;
        tile[tileBase + 3]    = (packed >> 24) & 0xFF;
    }

    // -- Left and right halo columns (scalar, 2 × tileH bytes) --
    if (tid < tileH) {
        int gr = gr0 + (int)tid;
        int gc = (int)gc0_interior - 1;
        tile[tid * tileW] =
            (gr >= 0 && (unsigned)gr < rows && gc >= 0 && (unsigned)gc < cols)
                ? src[(unsigned)gr * stride + (unsigned)gc]
                : 0;
    }
    if (tid >= tileH && tid < 2 * tileH) {
        unsigned int haloIdx = tid - tileH;
        int gr               = gr0 + (int)haloIdx;
        unsigned int gc      = gc0_interior + HIP_BYTE_TILE_COLS;
        tile[haloIdx * tileW + tileW - 1] =
            (gr >= 0 && (unsigned)gr < rows && gc < cols)
                ? src[(unsigned)gr * stride + gc]
                : 0;
    }

    __syncthreads();

    // -- Compute --
    // Fast path: pack 4 results into uint32_t, single coalesced 4-byte store.
    // Slow path: byte-by-byte for the last tile column if cols % 256 != 0.
    // The branch is wavefront-uniform for all but the rightmost tile column,
    // so it's essentially free.
    unsigned int row     = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int baseCol = blockIdx.x * HIP_BYTE_TILE_COLS + threadIdx.x * 4;
    unsigned int tx      = threadIdx.x * 4 + 1;
    unsigned int ty      = threadIdx.y + 1;

    if (row < rows && baseCol + 3 < cols) {
        uint32_t result = 0;
        for (int k = 0; k < 4; k++)
            result |= (unsigned)golRule(tile, tileW, tx + k, ty) << (k * 8);
        *reinterpret_cast<uint32_t *>(&dst[row * stride + baseCol]) = result;
    } else if (row < rows && baseCol < cols) {
        for (int k = 0; k < 4 && baseCol + k < cols; k++)
            dst[row * stride + baseCol + k] = golRule(tile, tileW, tx + k, ty);
    }
}

HIPTileGameOfLife::HIPTileGameOfLife(Grid &grid)
    : GPUEngine(std::move(grid), hipOps()) {
}

CellKind HIPTileGameOfLife::getCellKind() const {
    return CellKind::Byte;
}

void HIPTileGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
    dim3 block(HIP_BYTE_BLOCK_X, HIP_BYTE_BLOCK_Y);
    dim3 grid((cols_ + HIP_BYTE_TILE_COLS - 1) / HIP_BYTE_TILE_COLS,
              (rows_ + HIP_BYTE_BLOCK_Y - 1) / HIP_BYTE_BLOCK_Y);
    size_t shmem =
        (HIP_BYTE_TILE_COLS + 2) * (HIP_BYTE_BLOCK_Y + 2) * sizeof(uint8_t);
    hipGolKernelTile<<<grid, block, shmem>>>(
        src, dst, static_cast<unsigned int>(rows_),
        static_cast<unsigned int>(cols_),
        static_cast<unsigned int>(deviceStride_));
    HIP_CHECK(hipGetLastError());
}

// ===========================================================================
// 3. Bit-packed kernel (hip-bitpack)
// ===========================================================================
//
// GPU port of the CPU bitpack engine, tuned for HIP wavefront dimensions.
// Same d_rowSum3 / d_sum9 full-adder arithmetic as the CUDA bitpack kernel,
// but the 2D thread grid is shaped for 64-thread wavefronts: each thread owns
// one uint64_t word.
//
// Shared memory tile with a 1-word halo on each side is loaded via a
// cooperative flat fill: all threads in the block participate, iterating over
// the tile elements in strided fashion until the entire tile (including halo)
// is populated.  After a __syncthreads() barrier, each thread shifts bits
// across word boundaries using its neighbours in the shared memory tile,
// computes the full-adder sum, and applies the GoL rule.
//
// Block 64x4 = 256 threads, tile 66x6 words, 3168 B shared memory.

__device__ void d_rowSum3(uint64_t L, uint64_t C, uint64_t R, uint64_t &s1,
                          uint64_t &s0) {
    uint64_t t = L ^ C;
    s0         = t ^ R;
    s1         = (L & C) | (t & R);
}

__device__ void d_sum9(uint64_t p1, uint64_t p0, uint64_t c1, uint64_t c0,
                       uint64_t n1, uint64_t n0, uint64_t &o3, uint64_t &o2,
                       uint64_t &o1, uint64_t &o0) {
    uint64_t t0    = p0 ^ c0;
    uint64_t carry = p0 & c0;
    uint64_t q     = p1 ^ c1;
    uint64_t t1    = q ^ carry;
    uint64_t t2    = (p1 & c1) | (q & carry);

    o0          = t0 ^ n0;
    carry       = t0 & n0;
    uint64_t q1 = t1 ^ n1;
    o1          = q1 ^ carry;
    carry       = (t1 & n1) | (q1 & carry);
    o2          = t2 ^ carry;
    o3          = t2 & carry;
}

static constexpr int HIP_BIT_BLOCK_X = 64;
static constexpr int HIP_BIT_BLOCK_Y = 4;

__global__ void hipGolBitPackKernel(const uint64_t *src, uint64_t *dst,
                                    unsigned int rows, unsigned int nw,
                                    uint64_t lastMask) {
    extern __shared__ uint64_t stile[];
    const unsigned int tileW = blockDim.x + 2; // 66
    const unsigned int tileH = blockDim.y + 2; // 6

    const int gr0 = blockIdx.y * blockDim.y - 1;
    const int gw0 = blockIdx.x * blockDim.x - 1;

    // Cooperative flat tile load — all threads participate
    const unsigned int tileSize  = tileW * tileH;
    const unsigned int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int blockSize = blockDim.x * blockDim.y;

    for (unsigned int i = tid; i < tileSize; i += blockSize) {
        int gr = gr0 + (int)(i / tileW);
        int gw = gw0 + (int)(i % tileW);
        stile[i] =
            (gr >= 0 && (unsigned)gr < rows && gw >= 0 && (unsigned)gw < nw)
                ? src[(unsigned)gr * nw + (unsigned)gw]
                : 0;
    }

    __syncthreads();

    unsigned int w   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

    if (row < rows && w < nw) {
        // Current row: shift bits across word boundaries using shared memory
        uint64_t cC = stile[ty * tileW + tx];
        uint64_t cL = (cC << 1) | (stile[ty * tileW + tx - 1] >> 63);
        uint64_t cR = (cC >> 1) | (stile[ty * tileW + tx + 1] << 63);

        // Previous row
        uint64_t pC = stile[(ty - 1) * tileW + tx];
        uint64_t pL = (pC << 1) | (stile[(ty - 1) * tileW + tx - 1] >> 63);
        uint64_t pR = (pC >> 1) | (stile[(ty - 1) * tileW + tx + 1] << 63);

        // Next row
        uint64_t nC = stile[(ty + 1) * tileW + tx];
        uint64_t nL = (nC << 1) | (stile[(ty + 1) * tileW + tx - 1] >> 63);
        uint64_t nR = (nC >> 1) | (stile[(ty + 1) * tileW + tx + 1] << 63);

        uint64_t p1, p0, c1, c0, n1, n0;
        d_rowSum3(pL, pC, pR, p1, p0);
        d_rowSum3(cL, cC, cR, c1, c0);
        d_rowSum3(nL, nC, nR, n1, n0);

        uint64_t o3, o2, o1, o0;
        d_sum9(p1, p0, c1, c0, n1, n0, o3, o2, o1, o0);

        uint64_t is3    = ~o3 & ~o2 & o1 & o0;
        uint64_t is4    = ~o3 & o2 & ~o1 & ~o0;
        uint64_t result = is3 | (cC & is4);

        if (w == nw - 1)
            result &= lastMask;
        dst[row * nw + w] = result;
    }
}

HIPBitPackGameOfLife::HIPBitPackGameOfLife(BitGrid &grid)
    : GPUEngine(std::move(grid), hipOps()) {
}

CellKind HIPBitPackGameOfLife::getCellKind() const {
    return CellKind::BitPacked;
}

void HIPBitPackGameOfLife::launchKernel(const uint64_t *src, uint64_t *dst) {
    dim3 block(HIP_BIT_BLOCK_X, HIP_BIT_BLOCK_Y);
    dim3 grid((deviceStride_ + HIP_BIT_BLOCK_X - 1) / HIP_BIT_BLOCK_X,
              (rows_ + HIP_BIT_BLOCK_Y - 1) / HIP_BIT_BLOCK_Y);
    size_t shmem =
        (HIP_BIT_BLOCK_X + 2) * (HIP_BIT_BLOCK_Y + 2) * sizeof(uint64_t);

    uint64_t lastMask =
        (cols_ % 64 == 0) ? ~uint64_t(0) : (uint64_t(1) << (cols_ % 64)) - 1;

    hipGolBitPackKernel<<<grid, block, shmem>>>(
        src, dst, static_cast<unsigned int>(rows_),
        static_cast<unsigned int>(deviceStride_), lastMask);
    HIP_CHECK(hipGetLastError());
}

// ===========================================================================
// Test wrapper functions — single-step kernel execution on host data
// ===========================================================================

// Compute padded stride for byte buffers (multiple of 4).
static size_t padStride(size_t cols) {
    return (cols + 3) & ~size_t(3);
}

void hipSimpleKernelStep(const uint8_t *in, uint8_t *out, size_t rows,
                         size_t cols) {
    size_t stride = padStride(cols);
    size_t bytes  = rows * stride;
    uint8_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemset(d_in, 0, bytes));
    HIP_CHECK(hipMemset(d_out, 0, bytes));
    HIP_CHECK(
        hipMemcpy2D(d_in, stride, in, cols, cols, rows, hipMemcpyHostToDevice));

    unsigned int r = static_cast<unsigned int>(rows);
    unsigned int c = static_cast<unsigned int>(cols);
    unsigned int s = static_cast<unsigned int>(stride);

    unsigned int intCols =
        ((c > 2) ? ((c - 2) / HIP_SIMPLE_BLOCK_X) * HIP_SIMPLE_BLOCK_X : 0);
    unsigned int intRows =
        ((c > 2 && r > 2) ? ((r - 2) / HIP_SIMPLE_BLOCK_Y) * HIP_SIMPLE_BLOCK_Y
                          : 0);

    if (intCols > 0 && intRows > 0) {
        dim3 block(HIP_SIMPLE_BLOCK_X, HIP_SIMPLE_BLOCK_Y);
        dim3 grid(intCols / HIP_SIMPLE_BLOCK_X, intRows / HIP_SIMPLE_BLOCK_Y);
        hipGolKernelSimpleInterior<<<grid, block>>>(d_in, d_out, c, s);
        HIP_CHECK(hipGetLastError());
    }

    unsigned int perim       = 2 * c + 2 * (r - 2);
    unsigned int rightW      = (c >= 2 + intCols) ? (c - 2 - intCols) : 0;
    unsigned int rightCells  = (r >= 2) ? rightW * (r - 2) : 0;
    unsigned int bottomH     = (r >= 2 + intRows) ? (r - 2 - intRows) : 0;
    unsigned int bottomCells = bottomH * intCols;
    unsigned int totalBorder = perim + rightCells + bottomCells;

    if (totalBorder > 0) {
        int bThreads = 256;
        int bBlocks  = (totalBorder + bThreads - 1) / bThreads;
        hipGolKernelSimpleBorder<<<bBlocks, bThreads>>>(d_in, d_out, r, c, s,
                                                        intCols, intRows);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy2D(out, cols, d_out, stride, cols, rows,
                          hipMemcpyDeviceToHost));
    hipFree(d_in);
    hipFree(d_out);
}

void hipTileKernelStep(const uint8_t *in, uint8_t *out, size_t rows,
                       size_t cols) {
    size_t stride = padStride(cols);
    size_t bytes  = rows * stride;
    uint8_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemset(d_in, 0, bytes));
    HIP_CHECK(hipMemset(d_out, 0, bytes));
    HIP_CHECK(
        hipMemcpy2D(d_in, stride, in, cols, cols, rows, hipMemcpyHostToDevice));

    dim3 block(HIP_BYTE_BLOCK_X, HIP_BYTE_BLOCK_Y);
    dim3 grid((static_cast<unsigned int>(cols) + HIP_BYTE_TILE_COLS - 1) /
                  HIP_BYTE_TILE_COLS,
              (static_cast<unsigned int>(rows) + HIP_BYTE_BLOCK_Y - 1) /
                  HIP_BYTE_BLOCK_Y);
    size_t shmem =
        (HIP_BYTE_TILE_COLS + 2) * (HIP_BYTE_BLOCK_Y + 2) * sizeof(uint8_t);
    hipGolKernelTile<<<grid, block, shmem>>>(
        d_in, d_out, static_cast<unsigned int>(rows),
        static_cast<unsigned int>(cols), static_cast<unsigned int>(stride));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy2D(out, cols, d_out, stride, cols, rows,
                          hipMemcpyDeviceToHost));
    hipFree(d_in);
    hipFree(d_out);
}

void hipBitPackKernelStep(const uint64_t *in, uint64_t *out, size_t rows,
                          size_t stride, size_t cols) {
    size_t bytes = rows * stride * sizeof(uint64_t);
    uint64_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, in, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    dim3 block(HIP_BIT_BLOCK_X, HIP_BIT_BLOCK_Y);
    dim3 grid((static_cast<unsigned int>(stride) + HIP_BIT_BLOCK_X - 1) /
                  HIP_BIT_BLOCK_X,
              (static_cast<unsigned int>(rows) + HIP_BIT_BLOCK_Y - 1) /
                  HIP_BIT_BLOCK_Y);
    size_t shmem =
        (HIP_BIT_BLOCK_X + 2) * (HIP_BIT_BLOCK_Y + 2) * sizeof(uint64_t);

    uint64_t lastMask =
        (cols % 64 == 0) ? ~uint64_t(0) : (uint64_t(1) << (cols % 64)) - 1;

    hipGolBitPackKernel<<<grid, block, shmem>>>(
        d_in, d_out, static_cast<unsigned int>(rows),
        static_cast<unsigned int>(stride), lastMask);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out, d_out, bytes, hipMemcpyDeviceToHost));
    hipFree(d_in);
    hipFree(d_out);
}
