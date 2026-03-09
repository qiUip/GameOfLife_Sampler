#include "gol_gpu.h"

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static void cudaAlloc(void **ptr, size_t bytes) {
    CUDA_CHECK(cudaMalloc(ptr, bytes));
}
static void cudaFreeWrap(void *ptr) {
    cudaFree(ptr);
}
static void cudaCopyH2D(void *dst, const void *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
static void cudaCopyD2H(void *dst, const void *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
static void cudaSyncWrap() {
    CUDA_CHECK(cudaDeviceSynchronize());
}
static void cudaCheckLastWrap(const char *ctx) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", ctx,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static GpuOps cudaOps() {
    return {cudaAlloc,   cudaFreeWrap, cudaCopyH2D,
            cudaCopyD2H, cudaSyncWrap, cudaCheckLastWrap};
}

// ===========================================================================
// Byte-per-cell kernels — two engines
// ===========================================================================
//
//   1. golKernelSimple — one thread per cell, reads neighbours directly
//                        from global memory.  No shared memory.
//   2. golKernelTile   — shared memory tiling (128 × 8 tiles) with uint32_t
//                        vectorised loads and stores.
//
// Both kernels use the same branch-free GoL rule.

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

// -- 1. Simple kernel (cuda-simple) ---------------------------------------
//
// The most direct GPU translation of the CPU approach.
//
// On the CPU, `#pragma omp parallel for` dispatches rows to threads, and
// each thread walks columns left-to-right.  Here we go one step further:
// one thread per cell, no loops at all.  The kernel grid covers the entire
// domain — each thread computes exactly one output cell by reading its 8
// neighbours from global memory.
//
// Block: 32 × 8 = 256 threads.  The 32-thread warp dimension is along
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
// conditionals per-cell (8 branches × 2.5 billion cells).  The tiled
// kernels (§2–§3) don't need it — their boundary checks are per-tile
// (once per 256-thread block), so the cost of a second kernel launch
// outweighs the savings.
//
// Each cell in the grid may be read from global memory up to 9 times
// (once as itself, up to 8 times as a neighbour of adjacent cells).
// On a CPU, the hardware cache absorbs this reuse transparently —
// three active rows fit in L2, and the prefetcher keeps them warm.
// On a GPU, the per-SM L1 cache is small (typically 32-128 KB on modern
// GPUs) and shared across many concurrent warps on the same SM, so cache
// lines are often evicted before they can be reused.
//
// The tiled kernel (§2) fixes this by loading each cell exactly once into
// shared memory — a programmer-managed on-chip store that is private to
// each block and not subject to eviction by other blocks' traffic.

static constexpr int SIMPLE_BLOCK_X = 32;
static constexpr int SIMPLE_BLOCK_Y = 8;

// Interior kernel — zero conditionals.  Every thread is guaranteed to map
// to a valid interior cell (row in [1, rows-2], col in [1, cols-2]).
// The launch grid is rounded down to full blocks only.
__global__ void golKernelSimpleInterior(const uint8_t *src, uint8_t *dst,
                                        unsigned int cols) {
    unsigned int col = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = row * cols + col;

    unsigned int up = idx - cols;
    unsigned int dn = idx + cols;
    unsigned int nb = src[up - 1] + src[up] + src[up + 1] + src[idx - 1] +
                      src[idx + 1] + src[dn - 1] + src[dn] + src[dn + 1];

    unsigned int alive = src[idx];
    dst[idx]           = (nb == 3) | (alive & (nb == 2));
}

// Border kernel — handles the 4 perimeter strips plus any interior cells
// not covered by the full-block interior launch (partial-block remainders
// at the right and bottom edges).  Cell count is negligible so the branches
// have no measurable impact.
__global__ void golKernelSimpleBorder(const uint8_t *src, uint8_t *dst,
                                      unsigned int rows, unsigned int cols,
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

    unsigned int idx = row * cols + col;
    unsigned int nb  = 0;
    bool hasUp       = row > 0;
    bool hasDown     = row < rows - 1;
    bool hasLeft     = col > 0;
    bool hasRight    = col < cols - 1;

    if (hasUp) {
        unsigned int up = idx - cols;
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
        unsigned int dn = idx + cols;
        if (hasLeft)
            nb += src[dn - 1];
        nb += src[dn];
        if (hasRight)
            nb += src[dn + 1];
    }

    unsigned int alive = src[idx];
    dst[idx]           = (nb == 3) | (alive & (nb == 2));
}

CUDASimpleGameOfLife::CUDASimpleGameOfLife(Grid &grid)
    : GPUEngine(std::move(grid), cudaOps()) {
}

CellKind CUDASimpleGameOfLife::getCellKind() const {
    return CellKind::Byte;
}

void CUDASimpleGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
    unsigned int r = static_cast<unsigned int>(rows_);
    unsigned int c = static_cast<unsigned int>(cols_);

    // Interior: only full blocks — every thread maps to a valid cell.
    unsigned int intCols = ((c - 2) / SIMPLE_BLOCK_X) * SIMPLE_BLOCK_X;
    unsigned int intRows = ((r - 2) / SIMPLE_BLOCK_Y) * SIMPLE_BLOCK_Y;

    if (intCols > 0 && intRows > 0) {
        dim3 block(SIMPLE_BLOCK_X, SIMPLE_BLOCK_Y);
        dim3 grid(intCols / SIMPLE_BLOCK_X, intRows / SIMPLE_BLOCK_Y);
        golKernelSimpleInterior<<<grid, block>>>(src, dst, c);
        CUDA_CHECK(cudaGetLastError());
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
        golKernelSimpleBorder<<<bBlocks, bThreads>>>(src, dst, r, c, intCols,
                                                     intRows);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -- 2. Tiled kernel (cuda-tile) ------------------------------------------
//
// The simple kernel reads each cell from global memory up to 9 times (as a
// neighbour of surrounding cells). Shared memory eliminates this redundancy:
// each block cooperatively loads its tile (including a 1-cell halo) from global
// memory into on-chip shared memory, then all neighbour reads come from shared
// memory at ~100× lower latency.
//
// Unlike the per-SM L1 cache, shared memory is explicitly managed and private
// to each block — data stays resident until the block is done with it,
// regardless of what other blocks on the same SM are doing.
//
// Block: 32 × 8 = 256 threads. Each thread computes 4 consecutive cells in the
// column direction, so a block covers 128 columns × 8 rows. Tile: (128 + 2) ×
// (8 + 2) = 1,300 bytes shared memory (incl. halo). Grid: 2D launch, one block
// per tile — all tiles run concurrently.
//
// Loads: uint32_t vectorised reads for the 128 interior tile columns (4 bytes
// at a time), with scalar reads for the 2 halo columns. 128 cols / 4 = 32 words
// per row × 10 rows = 320 words total. Division by 32 (power of 2) compiles to
// a shift in a flat byte-by-byte load. Within a warp, 32 threads each load one
// uint32_t from consecutive addresses → 128 contiguous bytes, perfectly
// coalesced with no warp-straddling.
//
// Stores: each thread packs its 4 results into a uint32_t and writes it with a
// single 4-byte store. Within a warp, 32 threads × 4 bytes = 128 contiguous
// bytes — exactly one cache line, perfectly coalesced. Byte stores cannot merge
// into a single wide transaction the way a uint32_t store does, so the packing
// is critical for write throughput.
//
// Boundary handling: the last tile column (when cols is not a multiple of 128)
// falls through to byte-by-byte load and store paths. This affects only 1 out
// of ceil(cols/128) tile columns, so the branch cost is negligible.
//
static constexpr int BYTE_BLOCK_X          = 32;
static constexpr int BYTE_BLOCK_Y          = 8;
static constexpr int BYTE_CELLS_PER_THREAD = 4;
static constexpr int BYTE_TILE_COLS =
    BYTE_BLOCK_X * BYTE_CELLS_PER_THREAD; // 128

__global__ void golKernelTile(const uint8_t *src, uint8_t *dst,
                              unsigned int rows, unsigned int cols) {
    extern __shared__ uint8_t tile[];
    const unsigned int tileW     = BYTE_TILE_COLS + 2; // 130
    const unsigned int tileH     = blockDim.y + 2;     // 10
    const unsigned int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int blockSize = blockDim.x * blockDim.y; // 256

    // Interior columns start at global column gc0_interior.
    // Halo left column is gc0_interior - 1, halo right is gc0_interior + 128.
    const unsigned int gc0_interior = blockIdx.x * BYTE_TILE_COLS;
    const int gr0                   = (int)(blockIdx.y * blockDim.y) - 1;

    // Interior load (uint32_t vectorised) --
    const unsigned int wordsPerRow = BYTE_TILE_COLS / 4;  // 32
    const unsigned int totalWords  = wordsPerRow * tileH; // 320
    for (unsigned int i = tid; i < totalWords; i += blockSize) {
        unsigned int tileRow = i >> 5; // i / 32
        unsigned int wordIdx = i & 31; // i % 32
        int gr               = gr0 + (int)tileRow;
        unsigned int gc      = gc0_interior + wordIdx * 4;
        uint32_t packed      = 0;
        if (gr >= 0 && (unsigned)gr < rows && gc + 3 < cols)
            packed = *reinterpret_cast<const uint32_t *>(
                &src[(unsigned)gr * cols + gc]);
        else if (gr >= 0 && (unsigned)gr < rows && gc < cols) {
            // Partial word at right edge — load available bytes
            for (unsigned int b = 0; b < 4 && gc + b < cols; b++)
                packed |= (uint32_t)src[(unsigned)gr * cols + gc + b]
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
                ? src[(unsigned)gr * cols + (unsigned)gc]
                : 0;
    }
    if (tid >= tileH && tid < 2 * tileH) {
        unsigned int haloIdx = tid - tileH;
        int gr               = gr0 + (int)haloIdx;
        unsigned int gc      = gc0_interior + BYTE_TILE_COLS;
        tile[haloIdx * tileW + tileW - 1] =
            (gr >= 0 && (unsigned)gr < rows && gc < cols)
                ? src[(unsigned)gr * cols + gc]
                : 0;
    }

    __syncthreads();

    // -- Compute --
    // Fast path: pack 4 results into uint32_t, single coalesced 4-byte store.
    // Slow path: byte-by-byte for the last tile column if cols % 128 != 0.
    // The branch is warp-uniform for all but the rightmost tile column,
    // so it's essentially free.
    unsigned int row     = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int baseCol = blockIdx.x * BYTE_TILE_COLS + threadIdx.x * 4;
    unsigned int tx      = threadIdx.x * 4 + 1;
    unsigned int ty      = threadIdx.y + 1;

    if (row < rows && baseCol + 3 < cols) {
        uint32_t result = 0;
        for (int k = 0; k < 4; k++)
            result |= (unsigned)golRule(tile, tileW, tx + k, ty) << (k * 8);
        *reinterpret_cast<uint32_t *>(&dst[row * cols + baseCol]) = result;
    } else if (row < rows && baseCol < cols) {
        for (int k = 0; k < 4 && baseCol + k < cols; k++)
            dst[row * cols + baseCol + k] = golRule(tile, tileW, tx + k, ty);
    }
}

CUDATileGameOfLife::CUDATileGameOfLife(Grid &grid)
    : GPUEngine(std::move(grid), cudaOps()) {
}

CellKind CUDATileGameOfLife::getCellKind() const {
    return CellKind::Byte;
}

void CUDATileGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
    dim3 block(BYTE_BLOCK_X, BYTE_BLOCK_Y);
    dim3 grid((cols_ + BYTE_TILE_COLS - 1) / BYTE_TILE_COLS,
              (rows_ + BYTE_BLOCK_Y - 1) / BYTE_BLOCK_Y);
    size_t shmem = (BYTE_TILE_COLS + 2) * (BYTE_BLOCK_Y + 2) * sizeof(uint8_t);
    golKernelTile<<<grid, block, shmem>>>(src, dst,
                                          static_cast<unsigned int>(rows_),
                                          static_cast<unsigned int>(cols_));
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// 3. Bit-packed kernel (cuda-bitpack)
// ===========================================================================

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

static constexpr int BIT_BLOCK_X = 32;
static constexpr int BIT_BLOCK_Y = 8;

__global__ void golBitPackKernel(const uint64_t *src, uint64_t *dst,
                                 unsigned int rows, unsigned int nw,
                                 uint64_t lastMask) {
    extern __shared__ uint64_t stile[];
    const unsigned int tileW = blockDim.x + 2; // 34
    const unsigned int tileH = blockDim.y + 2; // 10

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

CUDABitPackGameOfLife::CUDABitPackGameOfLife(BitGrid &grid)
    : GPUEngine(std::move(grid), cudaOps()) {
}

CellKind CUDABitPackGameOfLife::getCellKind() const {
    return CellKind::BitPacked;
}

void CUDABitPackGameOfLife::launchKernel(const uint64_t *src, uint64_t *dst) {
    dim3 block(BIT_BLOCK_X, BIT_BLOCK_Y);
    dim3 grid((stride_ + BIT_BLOCK_X - 1) / BIT_BLOCK_X,
              (rows_ + BIT_BLOCK_Y - 1) / BIT_BLOCK_Y);
    size_t shmem = (BIT_BLOCK_X + 2) * (BIT_BLOCK_Y + 2) * sizeof(uint64_t);

    uint64_t lastMask =
        (cols_ % 64 == 0) ? ~uint64_t(0) : (uint64_t(1) << (cols_ % 64)) - 1;

    golBitPackKernel<<<grid, block, shmem>>>(
        src, dst, static_cast<unsigned int>(rows_),
        static_cast<unsigned int>(stride_), lastMask);
    CUDA_CHECK(cudaGetLastError());
}
