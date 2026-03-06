#include "gol.h"

#include <cstdio>
#include <type_traits>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════════
// CUDAEngineBase — shared device buffer management
// ═══════════════════════════════════════════════════════════════════════════

template<typename CellT, typename HostGridT>
CUDAEngineBase<CellT, HostGridT>::CUDAEngineBase(HostGridT hostGrid)
    : hostGrid_(std::move(hostGrid)),
      stride_(hostGrid_.getStride()) {
  rows_ = hostGrid_.getNumRows();
  cols_ = hostGrid_.getNumCols();
  size_t totalBytes = rows_ * stride_ * sizeof(CellT);
  CUDA_CHECK(cudaMalloc(&d_current_, totalBytes));
  CUDA_CHECK(cudaMalloc(&d_next_, totalBytes));
  CUDA_CHECK(cudaMemcpy(d_current_, hostGrid_.getData(), totalBytes,
                         cudaMemcpyHostToDevice));
}

template<typename CellT, typename HostGridT>
CUDAEngineBase<CellT, HostGridT>::~CUDAEngineBase() {
  cudaFree(d_current_);
  cudaFree(d_next_);
}

template<typename CellT, typename HostGridT>
void CUDAEngineBase<CellT, HostGridT>::takeStep() {
  launchKernel(d_current_, d_next_);
  CellT *tmp = d_current_;
  d_current_ = d_next_;
  d_next_ = tmp;
}

template<typename CellT, typename HostGridT>
Grid CUDAEngineBase<CellT, HostGridT>::getGrid() const {
  HostGridT tmp(rows_, cols_);
  CUDA_CHECK(cudaMemcpy(tmp.getData(), d_current_,
                         rows_ * stride_ * sizeof(CellT),
                         cudaMemcpyDeviceToHost));
  if constexpr (std::is_same_v<HostGridT, Grid>)
    return tmp;
  else
    return tmp.toGrid();
}

template<typename CellT, typename HostGridT>
void *CUDAEngineBase<CellT, HostGridT>::getRowDataRaw(size_t row) {
  CUDA_CHECK(cudaMemcpy(hostGrid_.getRowData(row),
                         d_current_ + row * stride_,
                         stride_ * sizeof(CellT),
                         cudaMemcpyDeviceToHost));
  return hostGrid_.getRowData(row);
}

template<typename CellT, typename HostGridT>
size_t CUDAEngineBase<CellT, HostGridT>::getStride() const { return stride_; }

template<typename CellT, typename HostGridT>
void CUDAEngineBase<CellT, HostGridT>::commitBoundaries() {
  CUDA_CHECK(cudaMemcpy(d_current_,
                         hostGrid_.getRowData(0),
                         stride_ * sizeof(CellT),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_current_ + (rows_ - 1) * stride_,
                         hostGrid_.getRowData(rows_ - 1),
                         stride_ * sizeof(CellT),
                         cudaMemcpyHostToDevice));
}

template<typename CellT, typename HostGridT>
void CUDAEngineBase<CellT, HostGridT>::sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit instantiations
template class CUDAEngineBase<uint8_t, Grid>;
template class CUDAEngineBase<uint64_t, BitGrid>;

// ═══════════════════════════════════════════════════════════════════════════
// Byte-per-cell kernel + CUDAGameOfLife
// ═══════════════════════════════════════════════════════════════════════════

static constexpr int BYTE_BLOCK_X = 32;
static constexpr int BYTE_BLOCK_Y = 8;
static constexpr int BYTE_CELLS_PER_THREAD = 4;
static constexpr int BYTE_TILE_COLS = BYTE_BLOCK_X * BYTE_CELLS_PER_THREAD; // 128

__global__ void golKernel(const uint8_t *src, uint8_t *dst,
                          unsigned int rows, unsigned int cols) {
  extern __shared__ uint8_t tile[];
  const unsigned int tileW = BYTE_TILE_COLS + 2;              // 130
  const unsigned int tileRows = blockDim.y + 2;               // 10
  const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int blockSize = blockDim.x * blockDim.y;     // 256

  const unsigned int gc0 = blockIdx.x * BYTE_TILE_COLS;      // first interior col
  const int gr0 = (int)(blockIdx.y * blockDim.y) - 1;        // first halo row

  // ── Load interior: 32 uint32_t words per tile row × 10 rows = 320 words ──
  const unsigned int wordsPerRow = BYTE_BLOCK_X;              // 32
  const unsigned int totalWords = wordsPerRow * tileRows;     // 320

  for (unsigned int i = tid; i < totalWords; i += blockSize) {
    unsigned int tileRow = i / wordsPerRow;
    unsigned int wordIdx = i % wordsPerRow;
    int gr = gr0 + (int)tileRow;
    unsigned int gc = gc0 + wordIdx * 4;
    unsigned int tileBase = tileRow * tileW + 1 + wordIdx * 4;

    if (gr >= 0 && (unsigned)gr < rows && gc + 3 < cols) {
      // Full word — vectorised uint32_t load (coalesced: 128 bytes per warp)
      uint32_t packed = *reinterpret_cast<const uint32_t*>(
          &src[(unsigned)gr * cols + gc]);
      tile[tileBase + 0] = packed;
      tile[tileBase + 1] = packed >> 8;
      tile[tileBase + 2] = packed >> 16;
      tile[tileBase + 3] = packed >> 24;
    } else {
      // Partial word or out-of-bounds — per-byte fallback
      for (int k = 0; k < 4; k++) {
        tile[tileBase + k] =
            (gr >= 0 && (unsigned)gr < rows && gc + k < cols)
              ? src[(unsigned)gr * cols + gc + k] : 0;
      }
    }
  }

  // ── Load left/right halo columns (scalar, 20 bytes total) ──
  if (tid < tileRows) {
    int gr = gr0 + (int)tid;
    int gc = (int)gc0 - 1;
    tile[tid * tileW] =
        (gr >= 0 && (unsigned)gr < rows && gc >= 0 && (unsigned)gc < cols)
          ? src[(unsigned)gr * cols + (unsigned)gc] : 0;
  }
  if (tid >= tileRows && tid < tileRows * 2) {
    unsigned int r = tid - tileRows;
    int gr = gr0 + (int)r;
    unsigned int gc = gc0 + BYTE_TILE_COLS;
    tile[r * tileW + tileW - 1] =
        (gr >= 0 && (unsigned)gr < rows && gc < cols)
          ? src[(unsigned)gr * cols + gc] : 0;
  }

  __syncthreads();

  // ── Compute: each thread processes 4 consecutive cells ──
  unsigned int baseCol = gc0 + threadIdx.x * 4;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int tx = threadIdx.x * 4 + 1;   // +1 for left halo offset
  unsigned int ty = threadIdx.y + 1;        // +1 for top halo offset

  if (row < rows && baseCol + 3 < cols) {
    // Vectorised path: all 4 cells valid
    uint32_t result = 0;
    for (int k = 0; k < 4; k++) {
      unsigned int cx = tx + k;
      unsigned int nb = tile[(ty-1)*tileW + cx-1] + tile[(ty-1)*tileW + cx] + tile[(ty-1)*tileW + cx+1]
                      + tile[ty*tileW + cx-1]                                + tile[ty*tileW + cx+1]
                      + tile[(ty+1)*tileW + cx-1] + tile[(ty+1)*tileW + cx] + tile[(ty+1)*tileW + cx+1];
      unsigned int alive = tile[ty * tileW + cx];
      unsigned int cell = (nb == 3) | (alive & (nb == 2));
      result |= (cell & 0xFF) << (k * 8);
    }
    *reinterpret_cast<uint32_t*>(&dst[row * cols + baseCol]) = result;
  } else if (row < rows && baseCol < cols) {
    // Scalar fallback: partial word at right edge
    for (int k = 0; k < 4 && baseCol + k < cols; k++) {
      unsigned int cx = tx + k;
      unsigned int nb = tile[(ty-1)*tileW + cx-1] + tile[(ty-1)*tileW + cx] + tile[(ty-1)*tileW + cx+1]
                      + tile[ty*tileW + cx-1]                                + tile[ty*tileW + cx+1]
                      + tile[(ty+1)*tileW + cx-1] + tile[(ty+1)*tileW + cx] + tile[(ty+1)*tileW + cx+1];
      unsigned int alive = tile[ty * tileW + cx];
      dst[row * cols + baseCol + k] = (nb == 3) | (alive & (nb == 2));
    }
  }
}

CUDAGameOfLife::CUDAGameOfLife(Grid &grid)
    : CUDAEngineBase(std::move(grid)) {}

CellKind CUDAGameOfLife::getCellKind() const { return CellKind::Byte; }

void CUDAGameOfLife::launchKernel(const uint8_t *src, uint8_t *dst) {
  dim3 block(BYTE_BLOCK_X, BYTE_BLOCK_Y);
  dim3 grid((cols_ + BYTE_TILE_COLS - 1) / BYTE_TILE_COLS,
            (rows_ + BYTE_BLOCK_Y - 1) / BYTE_BLOCK_Y);
  size_t shmem = (BYTE_TILE_COLS + 2) * (BYTE_BLOCK_Y + 2) * sizeof(uint8_t);
  golKernel<<<grid, block, shmem>>>(src, dst,
      static_cast<unsigned int>(rows_), static_cast<unsigned int>(cols_));
  CUDA_CHECK(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// Bit-packed kernel + CUDABitPackGameOfLife
// ═══════════════════════════════════════════════════════════════════════════

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

static constexpr int BIT_BLOCK_X = 32;
static constexpr int BIT_BLOCK_Y = 8;

__global__ void golBitPackKernel(const uint64_t *src, uint64_t *dst,
                                 unsigned int rows, unsigned int nw,
                                 uint64_t lastMask) {
  extern __shared__ uint64_t stile[];
  const unsigned int tileW = blockDim.x + 2;
  const unsigned int tileSize = tileW * (blockDim.y + 2);
  const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int blockSize = blockDim.x * blockDim.y;

  const int gr0 = blockIdx.y * blockDim.y - 1;
  const int gw0 = blockIdx.x * blockDim.x - 1;

  // Cooperative flat tile load — all threads participate, no divergent branches
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

    uint64_t is3 = ~o3 & ~o2 &  o1 & o0;
    uint64_t is4 = ~o3 &  o2 & ~o1 & ~o0;
    uint64_t result = is3 | (cC & is4);

    if (w == nw - 1) result &= lastMask;
    dst[row * nw + w] = result;
  }
}

CUDABitPackGameOfLife::CUDABitPackGameOfLife(Grid &grid)
    : CUDAEngineBase(BitGrid(grid)) {}

CUDABitPackGameOfLife::CUDABitPackGameOfLife(BitGrid &grid)
    : CUDAEngineBase(std::move(grid)) {}

CellKind CUDABitPackGameOfLife::getCellKind() const {
  return CellKind::BitPacked;
}

void CUDABitPackGameOfLife::launchKernel(const uint64_t *src, uint64_t *dst) {
  dim3 block(BIT_BLOCK_X, BIT_BLOCK_Y);
  dim3 grid((stride_ + BIT_BLOCK_X - 1) / BIT_BLOCK_X,
            (rows_ + BIT_BLOCK_Y - 1) / BIT_BLOCK_Y);
  size_t shmem = (BIT_BLOCK_X + 2) * (BIT_BLOCK_Y + 2) * sizeof(uint64_t);

  uint64_t lastMask = (cols_ % 64 == 0) ? ~uint64_t(0)
                                         : (uint64_t(1) << (cols_ % 64)) - 1;

  golBitPackKernel<<<grid, block, shmem>>>(src, dst,
      static_cast<unsigned int>(rows_), static_cast<unsigned int>(stride_),
      lastMask);
  CUDA_CHECK(cudaGetLastError());
}
