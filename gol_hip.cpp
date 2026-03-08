#include "gol_gpu.h"

#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call)                                                        \
    do                                                                         \
    {                                                                          \
        hipError_t err = (call);                                               \
        if (err != hipSuccess)                                                 \
        {                                                                      \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    hipGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static GpuOps hipOps()
{
    return {[](void **ptr, size_t bytes) { HIP_CHECK(hipMalloc(ptr, bytes)); },
            [](void *ptr) { hipFree(ptr); },
            [](void *dst, const void *src, size_t bytes) {
                HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
            },
            [](void *dst, const void *src, size_t bytes) {
                HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
            },
            []() { HIP_CHECK(hipDeviceSynchronize()); },
            [](const char *ctx) {
                hipError_t err = hipGetLastError();
                if (err != hipSuccess)
                {
                    fprintf(stderr, "HIP error after %s: %s\n", ctx,
                            hipGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
            }};
}

// ═══════════════════════════════════════════════════════════════════════════
// Byte-per-cell kernels
// ═══════════════════════════════════════════════════════════════════════════

__device__ uint8_t golRule(const uint8_t *tile, unsigned int tileW,
                           unsigned int tx, unsigned int ty) {
  unsigned int nb = tile[(ty-1)*tileW + tx-1] + tile[(ty-1)*tileW + tx] + tile[(ty-1)*tileW + tx+1]
                  + tile[ ty   *tileW + tx-1]                            + tile[ ty   *tileW + tx+1]
                  + tile[(ty+1)*tileW + tx-1] + tile[(ty+1)*tileW + tx] + tile[(ty+1)*tileW + tx+1];
  unsigned int alive = tile[ty * tileW + tx];
  return (nb == 3) | (alive & (nb == 2));
}

// ── 1. Simple kernel (hip-simple) ────────────────────────────────────────
// Block 64×4 = 256 threads (wavefront-aligned).

static constexpr int HIP_SIMPLE_BLOCK_X = 64;
static constexpr int HIP_SIMPLE_BLOCK_Y = 4;

__global__ void hipGolKernelSimpleInterior(const uint8_t *src, uint8_t *dst,
                                            unsigned int cols) {
  unsigned int col = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = row * cols + col;

  unsigned int up = idx - cols;
  unsigned int dn = idx + cols;
  unsigned int nb = src[up - 1] + src[up] + src[up + 1]
                  + src[idx - 1]          + src[idx + 1]
                  + src[dn - 1] + src[dn] + src[dn + 1];

  unsigned int alive = src[idx];
  dst[idx] = (nb == 3) | (alive & (nb == 2));
}

__global__ void hipGolKernelSimpleBorder(const uint8_t *src, uint8_t *dst,
                                          unsigned int rows, unsigned int cols,
                                          unsigned int interiorCols,
                                          unsigned int interiorRows) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int perimeterCells = 2 * cols + 2 * (rows - 2);
  unsigned int rightStripW = (cols >= 2 + interiorCols) ? (cols - 2 - interiorCols) : 0;
  unsigned int rightStripCells = (rows >= 2) ? rightStripW * (rows - 2) : 0;
  unsigned int bottomStripH = (rows >= 2 + interiorRows) ? (rows - 2 - interiorRows) : 0;
  unsigned int bottomStripCells = bottomStripH * interiorCols;

  unsigned int totalCells = perimeterCells + rightStripCells + bottomStripCells;
  if (tid >= totalCells) return;

  unsigned int row, col;
  if (tid < perimeterCells) {
    if (tid < cols) {
      row = 0; col = tid;
    } else if (tid < 2 * cols) {
      row = rows - 1; col = tid - cols;
    } else {
      unsigned int side = tid - 2 * cols;
      if (side < rows - 2) {
        row = 1 + side; col = 0;
      } else {
        row = 1 + (side - (rows - 2)); col = cols - 1;
      }
    }
  } else if (tid < perimeterCells + rightStripCells) {
    unsigned int s = tid - perimeterCells;
    row = 1 + s / rightStripW;
    col = 1 + interiorCols + s % rightStripW;
  } else {
    unsigned int s = tid - perimeterCells - rightStripCells;
    row = 1 + interiorRows + s / interiorCols;
    col = 1 + s % interiorCols;
  }

  unsigned int idx = row * cols + col;
  unsigned int nb = 0;
  bool hasUp    = row > 0;
  bool hasDown  = row < rows - 1;
  bool hasLeft  = col > 0;
  bool hasRight = col < cols - 1;

  if (hasUp) {
    unsigned int up = idx - cols;
    if (hasLeft)  nb += src[up - 1];
                  nb += src[up];
    if (hasRight) nb += src[up + 1];
  }
  if (hasLeft)    nb += src[idx - 1];
  if (hasRight)   nb += src[idx + 1];
  if (hasDown) {
    unsigned int dn = idx + cols;
    if (hasLeft)  nb += src[dn - 1];
                  nb += src[dn];
    if (hasRight) nb += src[dn + 1];
  }

  unsigned int alive = src[idx];
  dst[idx] = (nb == 3) | (alive & (nb == 2));
}

HIPSimpleGameOfLife::HIPSimpleGameOfLife(Grid &grid)
    : GPUEngine(std::move(grid), hipOps()) {}

CellKind HIPSimpleGameOfLife::getCellKind() const { return CellKind::Byte; }

void HIPSimpleGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
  unsigned int r = static_cast<unsigned int>(rows_);
  unsigned int c = static_cast<unsigned int>(cols_);

  unsigned int intCols = ((c - 2) / HIP_SIMPLE_BLOCK_X) * HIP_SIMPLE_BLOCK_X;
  unsigned int intRows = ((r - 2) / HIP_SIMPLE_BLOCK_Y) * HIP_SIMPLE_BLOCK_Y;

  if (intCols > 0 && intRows > 0) {
    dim3 block(HIP_SIMPLE_BLOCK_X, HIP_SIMPLE_BLOCK_Y);
    dim3 grid(intCols / HIP_SIMPLE_BLOCK_X, intRows / HIP_SIMPLE_BLOCK_Y);
    hipGolKernelSimpleInterior<<<grid, block>>>(src, dst, c);
    HIP_CHECK(hipGetLastError());
  }

  unsigned int perim = 2 * c + 2 * (r - 2);
  unsigned int rightW = (c >= 2 + intCols) ? (c - 2 - intCols) : 0;
  unsigned int rightCells = (r >= 2) ? rightW * (r - 2) : 0;
  unsigned int bottomH = (r >= 2 + intRows) ? (r - 2 - intRows) : 0;
  unsigned int bottomCells = bottomH * intCols;
  unsigned int totalBorder = perim + rightCells + bottomCells;

  if (totalBorder > 0) {
    int bThreads = 256;
    int bBlocks = (totalBorder + bThreads - 1) / bThreads;
    hipGolKernelSimpleBorder<<<bBlocks, bThreads>>>(src, dst, r, c,
                                                     intCols, intRows);
    HIP_CHECK(hipGetLastError());
  }
}

// ── 2. Tiled kernel (hip-tile) ───────────────────────────────────────────
// Block 64×8 = 512 threads, tile 256×8, shmem 2580B.
// wordsPerRow = 64 (power of 2 — shift instead of division).

static constexpr int HIP_BYTE_BLOCK_X = 64;
static constexpr int HIP_BYTE_BLOCK_Y = 8;
static constexpr int HIP_BYTE_CELLS_PER_THREAD = 4;
static constexpr int HIP_BYTE_TILE_COLS = HIP_BYTE_BLOCK_X * HIP_BYTE_CELLS_PER_THREAD; // 256

__global__ void hipGolKernelTile(const uint8_t *src, uint8_t *dst,
                                  unsigned int rows, unsigned int cols) {
  extern __shared__ uint8_t tile[];
  const unsigned int tileW     = HIP_BYTE_TILE_COLS + 2;           // 258
  const unsigned int tileH     = blockDim.y + 2;                   // 10
  const unsigned int tid       = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int blockSize = blockDim.x * blockDim.y;          // 512

  const unsigned int gc0_interior = blockIdx.x * HIP_BYTE_TILE_COLS;
  const int gr0 = (int)(blockIdx.y * blockDim.y) - 1;

  // Interior load (uint32_t vectorised)
  const unsigned int wordsPerRow = HIP_BYTE_TILE_COLS / 4;  // 64
  const unsigned int totalWords  = wordsPerRow * tileH;      // 640
  for (unsigned int i = tid; i < totalWords; i += blockSize) {
    unsigned int tileRow = i >> 6;        // i / 64
    unsigned int wordIdx = i & 63;        // i % 64
    int gr = gr0 + (int)tileRow;
    unsigned int gc = gc0_interior + wordIdx * 4;
    uint32_t packed = 0;
    if (gr >= 0 && (unsigned)gr < rows && gc + 3 < cols)
      packed = *reinterpret_cast<const uint32_t*>(&src[(unsigned)gr * cols + gc]);
    else if (gr >= 0 && (unsigned)gr < rows && gc < cols) {
      for (unsigned int b = 0; b < 4 && gc + b < cols; b++)
        packed |= (uint32_t)src[(unsigned)gr * cols + gc + b] << (b * 8);
    }
    unsigned int tileBase = tileRow * tileW + 1 + wordIdx * 4;
    tile[tileBase + 0] = packed & 0xFF;
    tile[tileBase + 1] = (packed >> 8) & 0xFF;
    tile[tileBase + 2] = (packed >> 16) & 0xFF;
    tile[tileBase + 3] = (packed >> 24) & 0xFF;
  }

  // Left and right halo columns
  if (tid < tileH) {
    int gr = gr0 + (int)tid;
    int gc = (int)gc0_interior - 1;
    tile[tid * tileW] = (gr >= 0 && (unsigned)gr < rows && gc >= 0 && (unsigned)gc < cols)
                          ? src[(unsigned)gr * cols + (unsigned)gc] : 0;
  }
  if (tid >= tileH && tid < 2 * tileH) {
    unsigned int haloIdx = tid - tileH;
    int gr = gr0 + (int)haloIdx;
    unsigned int gc = gc0_interior + HIP_BYTE_TILE_COLS;
    tile[haloIdx * tileW + tileW - 1] = (gr >= 0 && (unsigned)gr < rows && gc < cols)
                                           ? src[(unsigned)gr * cols + gc] : 0;
  }

  __syncthreads();

  // Compute — pack 4 results into uint32_t for coalesced stores
  unsigned int row     = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int baseCol = blockIdx.x * HIP_BYTE_TILE_COLS + threadIdx.x * 4;
  unsigned int tx      = threadIdx.x * 4 + 1;
  unsigned int ty      = threadIdx.y + 1;

  if (row < rows && baseCol + 3 < cols) {
    uint32_t result = 0;
    for (int k = 0; k < 4; k++)
      result |= (unsigned)golRule(tile, tileW, tx + k, ty) << (k * 8);
    *reinterpret_cast<uint32_t*>(&dst[row * cols + baseCol]) = result;
  } else if (row < rows && baseCol < cols) {
    for (int k = 0; k < 4 && baseCol + k < cols; k++)
      dst[row * cols + baseCol + k] = golRule(tile, tileW, tx + k, ty);
  }
}

HIPTileGameOfLife::HIPTileGameOfLife(Grid &grid)
    : GPUEngine(std::move(grid), hipOps()) {}

CellKind HIPTileGameOfLife::getCellKind() const { return CellKind::Byte; }

void HIPTileGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
  dim3 block(HIP_BYTE_BLOCK_X, HIP_BYTE_BLOCK_Y);
  dim3 grid((cols_ + HIP_BYTE_TILE_COLS - 1) / HIP_BYTE_TILE_COLS,
            (rows_ + HIP_BYTE_BLOCK_Y - 1) / HIP_BYTE_BLOCK_Y);
  size_t shmem = (HIP_BYTE_TILE_COLS + 2) * (HIP_BYTE_BLOCK_Y + 2) * sizeof(uint8_t);
  hipGolKernelTile<<<grid, block, shmem>>>(src, dst,
      static_cast<unsigned int>(rows_), static_cast<unsigned int>(cols_));
  HIP_CHECK(hipGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Bit-packed kernel (hip-bitpack)
// ═══════════════════════════════════════════════════════════════════════════
// Block 64×4 = 256 threads, tile 66×6, shmem 3168B.

__device__ void d_rowSum3(uint64_t L, uint64_t C, uint64_t R,
                          uint64_t &s1, uint64_t &s0) {
  uint64_t t = L ^ C;
  s0 = t ^ R;
  s1 = (L & C) | (t & R);
}

__device__ void d_sum9(uint64_t p1, uint64_t p0,
                       uint64_t c1, uint64_t c0,
                       uint64_t n1, uint64_t n0,
                       uint64_t &o3, uint64_t &o2,
                       uint64_t &o1, uint64_t &o0) {
  uint64_t t0    = p0 ^ c0;
  uint64_t carry = p0 & c0;
  uint64_t q     = p1 ^ c1;
  uint64_t t1    = q ^ carry;
  uint64_t t2    = (p1 & c1) | (q & carry);

  o0    = t0 ^ n0;
  carry = t0 & n0;
  uint64_t q1 = t1 ^ n1;
  o1    = q1 ^ carry;
  carry = (t1 & n1) | (q1 & carry);
  o2    = t2 ^ carry;
  o3    = t2 & carry;
}

static constexpr int HIP_BIT_BLOCK_X = 64;
static constexpr int HIP_BIT_BLOCK_Y = 4;

__global__ void hipGolBitPackKernel(const uint64_t *src, uint64_t *dst,
                                     unsigned int rows, unsigned int nw,
                                     uint64_t lastMask) {
  extern __shared__ uint64_t stile[];
  const unsigned int tileW = blockDim.x + 2;  // 66
  const unsigned int tileH = blockDim.y + 2;  // 6

  const int gr0 = blockIdx.y * blockDim.y - 1;
  const int gw0 = blockIdx.x * blockDim.x - 1;

  const unsigned int tileSize  = tileW * tileH;
  const unsigned int tid       = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int blockSize = blockDim.x * blockDim.y;

  for (unsigned int i = tid; i < tileSize; i += blockSize) {
    int gr = gr0 + (int)(i / tileW);
    int gw = gw0 + (int)(i % tileW);
    stile[i] = (gr >= 0 && (unsigned)gr < rows && gw >= 0 && (unsigned)gw < nw)
                 ? src[(unsigned)gr * nw + (unsigned)gw] : 0;
  }

  __syncthreads();

  unsigned int w   = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

  if (row < rows && w < nw) {
    uint64_t cC = stile[ty * tileW + tx];
    uint64_t cL = (cC << 1) | (stile[ty * tileW + tx - 1] >> 63);
    uint64_t cR = (cC >> 1) | (stile[ty * tileW + tx + 1] << 63);

    uint64_t pC = stile[(ty - 1) * tileW + tx];
    uint64_t pL = (pC << 1) | (stile[(ty - 1) * tileW + tx - 1] >> 63);
    uint64_t pR = (pC >> 1) | (stile[(ty - 1) * tileW + tx + 1] << 63);

    uint64_t nC = stile[(ty + 1) * tileW + tx];
    uint64_t nL = (nC << 1) | (stile[(ty + 1) * tileW + tx - 1] >> 63);
    uint64_t nR = (nC >> 1) | (stile[(ty + 1) * tileW + tx + 1] << 63);

    uint64_t p1, p0, c1, c0, n1, n0;
    d_rowSum3(pL, pC, pR, p1, p0);
    d_rowSum3(cL, cC, cR, c1, c0);
    d_rowSum3(nL, nC, nR, n1, n0);

    uint64_t o3, o2, o1, o0;
    d_sum9(p1, p0, c1, c0, n1, n0, o3, o2, o1, o0);

    uint64_t is3 = ~o3 & ~o2 &  o1 & o0;
    uint64_t is4 = ~o3 &  o2 & ~o1 & ~o0;
    uint64_t result = is3 | (cC & is4);

    if (w == nw - 1) result &= lastMask;
    dst[row * nw + w] = result;
  }
}

HIPBitPackGameOfLife::HIPBitPackGameOfLife(Grid &grid)
    : GPUEngine(BitGrid(grid), hipOps()) {}

HIPBitPackGameOfLife::HIPBitPackGameOfLife(BitGrid &grid)
    : GPUEngine(std::move(grid), hipOps()) {}

CellKind HIPBitPackGameOfLife::getCellKind() const {
  return CellKind::BitPacked;
}

void HIPBitPackGameOfLife::launchKernel(const uint64_t *src, uint64_t *dst) {
  dim3 block(HIP_BIT_BLOCK_X, HIP_BIT_BLOCK_Y);
  dim3 grid((stride_ + HIP_BIT_BLOCK_X - 1) / HIP_BIT_BLOCK_X,
            (rows_ + HIP_BIT_BLOCK_Y - 1) / HIP_BIT_BLOCK_Y);
  size_t shmem = (HIP_BIT_BLOCK_X + 2) * (HIP_BIT_BLOCK_Y + 2) * sizeof(uint64_t);

  uint64_t lastMask = (cols_ % 64 == 0) ? ~uint64_t(0)
                                         : (uint64_t(1) << (cols_ % 64)) - 1;

  hipGolBitPackKernel<<<grid, block, shmem>>>(src, dst,
      static_cast<unsigned int>(rows_), static_cast<unsigned int>(stride_),
      lastMask);
  HIP_CHECK(hipGetLastError());
}
