#include "gol.h"

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ── Byte-per-cell kernel ────────────────────────────────────────────────────

static constexpr int BYTE_BLOCK_X = 32;
static constexpr int BYTE_BLOCK_Y = 8;

__global__ void golKernel(const uint8_t *src, uint8_t *dst,
                          size_t rows, size_t cols) {
  extern __shared__ uint8_t tile[];
  const int tileW = blockDim.x + 2;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

  // Load center cell
  tile[ty * tileW + tx] = (row < rows && col < cols)
                            ? src[row * cols + col] : 0;

  // Halo: edge threads load 1 extra cell
  if (threadIdx.x == 0)
    tile[ty * tileW] = (col > 0 && row < rows)
                         ? src[row * cols + col - 1] : 0;
  if (threadIdx.x == blockDim.x - 1)
    tile[ty * tileW + tx + 1] = (col + 1 < cols && row < rows)
                                  ? src[row * cols + col + 1] : 0;
  if (threadIdx.y == 0)
    tile[tx] = (row > 0 && col < cols)
                 ? src[(row - 1) * cols + col] : 0;
  if (threadIdx.y == blockDim.y - 1)
    tile[(ty + 1) * tileW + tx] = (row + 1 < rows && col < cols)
                                    ? src[(row + 1) * cols + col] : 0;

  // 4 corner halos
  if (threadIdx.x == 0 && threadIdx.y == 0)
    tile[0] = (row > 0 && col > 0)
                ? src[(row - 1) * cols + col - 1] : 0;
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
    tile[tx + 1] = (row > 0 && col + 1 < cols)
                     ? src[(row - 1) * cols + col + 1] : 0;
  if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
    tile[(ty + 1) * tileW] = (row + 1 < rows && col > 0)
                               ? src[(row + 1) * cols + col - 1] : 0;
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
    tile[(ty + 1) * tileW + tx + 1] = (row + 1 < rows && col + 1 < cols)
                                        ? src[(row + 1) * cols + col + 1] : 0;

  __syncthreads();

  if (row < rows && col < cols) {
    uint8_t nb = tile[(ty - 1) * tileW + tx - 1] + tile[(ty - 1) * tileW + tx]
               + tile[(ty - 1) * tileW + tx + 1]
               + tile[ty * tileW + tx - 1] + tile[ty * tileW + tx + 1]
               + tile[(ty + 1) * tileW + tx - 1] + tile[(ty + 1) * tileW + tx]
               + tile[(ty + 1) * tileW + tx + 1];
    uint8_t alive = tile[ty * tileW + tx];
    dst[row * cols + col] = (nb == 3) | (alive & (nb == 2));
  }
}

// ── CUDAGameOfLife implementation ───────────────────────────────────────────

CUDAGameOfLife::CUDAGameOfLife(Grid &grid)
    : hostGrid_(std::move(grid)) {
  rows_ = hostGrid_.getNumRows();
  cols_ = hostGrid_.getNumCols();
  size_t bytes = rows_ * cols_ * sizeof(uint8_t);
  CUDA_CHECK(cudaMalloc(&d_current_, bytes));
  CUDA_CHECK(cudaMalloc(&d_next_, bytes));
  CUDA_CHECK(cudaMemcpy(d_current_, hostGrid_.getData(), bytes,
                         cudaMemcpyHostToDevice));
}

CUDAGameOfLife::~CUDAGameOfLife() {
  cudaFree(d_current_);
  cudaFree(d_next_);
}

void CUDAGameOfLife::takeStep() {
  dim3 block(BYTE_BLOCK_X, BYTE_BLOCK_Y);
  dim3 grid((cols_ + BYTE_BLOCK_X - 1) / BYTE_BLOCK_X,
            (rows_ + BYTE_BLOCK_Y - 1) / BYTE_BLOCK_Y);
  size_t shmem = (BYTE_BLOCK_X + 2) * (BYTE_BLOCK_Y + 2) * sizeof(uint8_t);

  golKernel<<<grid, block, shmem>>>(d_current_, d_next_, rows_, cols_);
  CUDA_CHECK(cudaGetLastError());

  uint8_t *tmp = d_current_;
  d_current_ = d_next_;
  d_next_ = tmp;
}

Grid CUDAGameOfLife::getGrid() const {
  Grid result(rows_, cols_);
  CUDA_CHECK(cudaMemcpy(result.getData(), d_current_,
                         rows_ * cols_ * sizeof(uint8_t),
                         cudaMemcpyDeviceToHost));
  return result;
}

void *CUDAGameOfLife::getRowDataRaw(size_t row) {
  CUDA_CHECK(cudaMemcpy(hostGrid_.getRowData(row),
                         d_current_ + row * cols_,
                         cols_ * sizeof(uint8_t),
                         cudaMemcpyDeviceToHost));
  return hostGrid_.getRowData(row);
}

size_t CUDAGameOfLife::getStride() const { return cols_; }

CellKind CUDAGameOfLife::getCellKind() const { return CellKind::Byte; }

void CUDAGameOfLife::commitBoundaries() {
  // Copy ghost rows 0 and rows-1 from host back to device
  CUDA_CHECK(cudaMemcpy(d_current_,
                         hostGrid_.getRowData(0),
                         cols_ * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_current_ + (rows_ - 1) * cols_,
                         hostGrid_.getRowData(rows_ - 1),
                         cols_ * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
}

// ── Bit-packed kernel helpers ───────────────────────────────────────────────

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

// ── Bit-packed kernel ───────────────────────────────────────────────────────

static constexpr int BIT_BLOCK_X = 32;
static constexpr int BIT_BLOCK_Y = 8;

__global__ void golBitPackKernel(const uint64_t *src, uint64_t *dst,
                                 size_t rows, size_t nw, uint64_t lastMask) {
  extern __shared__ uint64_t stile[];
  const int tileW = blockDim.x + 2;

  int w   = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

  // Load center word
  stile[ty * tileW + tx] = (row < rows && w < nw) ? src[row * nw + w] : 0;

  // Halo: left/right word neighbors
  if (threadIdx.x == 0)
    stile[ty * tileW] = (w > 0 && row < rows) ? src[row * nw + w - 1] : 0;
  if (threadIdx.x == blockDim.x - 1)
    stile[ty * tileW + tx + 1] = (w + 1 < nw && row < rows)
                                   ? src[row * nw + w + 1] : 0;

  // Halo: top/bottom rows
  if (threadIdx.y == 0)
    stile[tx] = (row > 0 && w < nw) ? src[(row - 1) * nw + w] : 0;
  if (threadIdx.y == blockDim.y - 1)
    stile[(ty + 1) * tileW + tx] = (row + 1 < rows && w < nw)
                                     ? src[(row + 1) * nw + w] : 0;

  // Halo: top-left, top-right
  if (threadIdx.x == 0 && threadIdx.y == 0)
    stile[0] = (row > 0 && w > 0) ? src[(row - 1) * nw + w - 1] : 0;
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
    stile[tx + 1] = (row > 0 && w + 1 < nw) ? src[(row - 1) * nw + w + 1] : 0;

  // Halo: bottom-left, bottom-right
  if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
    stile[(ty + 1) * tileW] = (row + 1 < rows && w > 0)
                                ? src[(row + 1) * nw + w - 1] : 0;
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
    stile[(ty + 1) * tileW + tx + 1] = (row + 1 < rows && w + 1 < nw)
                                         ? src[(row + 1) * nw + w + 1] : 0;

  __syncthreads();

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

// ── CUDABitPackGameOfLife implementation ────────────────────────────────────

CUDABitPackGameOfLife::CUDABitPackGameOfLife(Grid &grid)
    : hostGrid_(grid),
      wordsPerRow_((grid.getNumCols() + 63) / 64) {
  rows_ = hostGrid_.getNumRows();
  cols_ = hostGrid_.getNumCols();
  size_t totalWords = rows_ * wordsPerRow_;
  CUDA_CHECK(cudaMalloc(&d_current_, totalWords * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_next_, totalWords * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_current_, hostGrid_.getData(),
                         totalWords * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));
}

CUDABitPackGameOfLife::CUDABitPackGameOfLife(BitGrid &grid)
    : hostGrid_(std::move(grid)),
      wordsPerRow_(hostGrid_.getStride()) {
  rows_ = hostGrid_.getNumRows();
  cols_ = hostGrid_.getNumCols();
  size_t totalWords = rows_ * wordsPerRow_;
  CUDA_CHECK(cudaMalloc(&d_current_, totalWords * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_next_, totalWords * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_current_, hostGrid_.getData(),
                         totalWords * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));
}

CUDABitPackGameOfLife::~CUDABitPackGameOfLife() {
  cudaFree(d_current_);
  cudaFree(d_next_);
}

void CUDABitPackGameOfLife::takeStep() {
  dim3 block(BIT_BLOCK_X, BIT_BLOCK_Y);
  dim3 grid((wordsPerRow_ + BIT_BLOCK_X - 1) / BIT_BLOCK_X,
            (rows_ + BIT_BLOCK_Y - 1) / BIT_BLOCK_Y);
  size_t shmem = (BIT_BLOCK_X + 2) * (BIT_BLOCK_Y + 2) * sizeof(uint64_t);

  uint64_t lastMask = (cols_ % 64 == 0) ? ~uint64_t(0)
                                         : (uint64_t(1) << (cols_ % 64)) - 1;

  golBitPackKernel<<<grid, block, shmem>>>(d_current_, d_next_,
                                            rows_, wordsPerRow_, lastMask);
  CUDA_CHECK(cudaGetLastError());

  uint64_t *tmp = d_current_;
  d_current_ = d_next_;
  d_next_ = tmp;
}

Grid CUDABitPackGameOfLife::getGrid() const {
  size_t totalWords = rows_ * wordsPerRow_;
  // Copy to a temporary BitGrid to convert
  BitGrid tmp(rows_, cols_);
  CUDA_CHECK(cudaMemcpy(tmp.getData(), d_current_,
                         totalWords * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  return tmp.toGrid();
}

void *CUDABitPackGameOfLife::getRowDataRaw(size_t row) {
  CUDA_CHECK(cudaMemcpy(hostGrid_.getRowData(row),
                         d_current_ + row * wordsPerRow_,
                         wordsPerRow_ * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  return hostGrid_.getRowData(row);
}

size_t CUDABitPackGameOfLife::getStride() const { return wordsPerRow_; }

CellKind CUDABitPackGameOfLife::getCellKind() const {
  return CellKind::BitPacked;
}

void CUDABitPackGameOfLife::commitBoundaries() {
  // Copy ghost rows 0 and rows-1 from host back to device
  CUDA_CHECK(cudaMemcpy(d_current_,
                         hostGrid_.getRowData(0),
                         wordsPerRow_ * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_current_ + (rows_ - 1) * wordsPerRow_,
                         hostGrid_.getRowData(rows_ - 1),
                         wordsPerRow_ * sizeof(uint64_t),
                         cudaMemcpyHostToDevice));
}
