#include "gol.h"

#include <cstdlib>
#include <omp.h>

// ── BitGrid implementation ──────────────────────────────────────────────────

BitGrid::BitGrid(size_t rows, size_t cols)
    : GridStorage(rows, cols, (cols + 63) / 64),
      wordsPerRow_((cols + 63) / 64) {}

BitGrid::BitGrid(const Grid &g)
    : GridStorage(g.getNumRows(), g.getNumCols(), (g.getNumCols() + 63) / 64),
      wordsPerRow_((g.getNumCols() + 63) / 64) {
  for (size_t r = 0; r < rows_; ++r)
    for (size_t c = 0; c < cols_; ++c)
      if (g.getCell(r, c))
        data_[r * wordsPerRow_ + c / 64] |= uint64_t(1) << (c % 64);
}

Grid BitGrid::toGrid() const {
  Grid g(rows_, cols_);
  for (size_t r = 0; r < rows_; ++r)
    for (size_t c = 0; c < cols_; ++c)
      g.setCell(r, c, (data_[r * wordsPerRow_ + c / 64] >> (c % 64)) & 1);
  return g;
}

size_t BitGrid::aliveCells() const {
  size_t count = 0;
  for (size_t i = 0; i < rows_ * wordsPerRow_; ++i)
    count += __builtin_popcountll(data_[i]);
  return count;
}

// ── Bit-parallel helpers ─────────────────────────────────────────────────────

// rowSum3: 3-input per-bit adder -> 2-bit result (s1:s0) per bit position.
static inline void rowSum3(uint64_t L, uint64_t C, uint64_t R,
                            uint64_t &s1, uint64_t &s0) {
  uint64_t t = L ^ C;
  s0 = t ^ R;
  s1 = (L & C) | (t & R);
}

// sum9: add three 2-bit row-sums to a 4-bit result per bit position.
static inline void sum9(uint64_t p1, uint64_t p0,
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

// ── BitPackGameOfLife implementation ────────────────────────────────────────

BitPackGameOfLife::BitPackGameOfLife(Grid &grid)
    : current_(grid),
      next_(grid.getNumRows(), grid.getNumCols()),
      wordsPerRow_((grid.getNumCols() + 63) / 64) {
  rows_ = grid.getNumRows();
  cols_ = grid.getNumCols();
}

BitPackGameOfLife::BitPackGameOfLife(BitGrid &grid)
    : current_(std::move(grid)),
      next_(current_.getNumRows(), current_.getNumCols()),
      wordsPerRow_(current_.getStride()) {
  rows_ = current_.getNumRows();
  cols_ = current_.getNumCols();
}

void BitPackGameOfLife::takeStep() {
  const size_t    nw  = wordsPerRow_;
  const uint64_t *src = current_.getData();
  uint64_t       *dst = next_.getData();

  const uint64_t lastMask = (cols_ % 64 == 0) ? ~uint64_t(0)
                                               : (uint64_t(1) << (cols_ % 64)) - 1;

  uint64_t *zeroRow = static_cast<uint64_t *>(std::calloc(nw, sizeof(uint64_t)));

#pragma omp parallel for schedule(static)
  for (size_t row = 0; row < rows_; ++row) {
    const uint64_t *p = (row > 0)         ? src + (row - 1) * nw : zeroRow;
    const uint64_t *c =                     src +  row      * nw;
    const uint64_t *n = (row < rows_ - 1) ? src + (row + 1) * nw : zeroRow;
    uint64_t       *o =                     dst +  row      * nw;

    for (size_t w = 0; w < nw; ++w) {
      const uint64_t cC = c[w];
      const uint64_t cL = (cC << 1) | (w > 0      ? c[w-1] >> 63 : 0ULL);
      const uint64_t cR = (cC >> 1) | (w < nw - 1 ? c[w+1] << 63 : 0ULL);
      uint64_t pC=p[w], pL=(pC<<1)|(w>0 ? p[w-1]>>63 : 0ULL),
                        pR=(pC>>1)|(w<nw-1 ? p[w+1]<<63 : 0ULL);
      uint64_t nC=n[w], nL=(nC<<1)|(w>0 ? n[w-1]>>63 : 0ULL),
                        nR=(nC>>1)|(w<nw-1 ? n[w+1]<<63 : 0ULL);
      uint64_t p1,p0,c1,c0,n1,n0;
      rowSum3(pL,pC,pR,p1,p0);
      rowSum3(cL,cC,cR,c1,c0);
      rowSum3(nL,nC,nR,n1,n0);
      uint64_t o3,o2,o1,o0;
      sum9(p1,p0,c1,c0,n1,n0,o3,o2,o1,o0);
      const uint64_t is3 = ~o3 & ~o2 &  o1 & o0;
      const uint64_t is4 = ~o3 &  o2 & ~o1 & ~o0;
      o[w] = is3 | (cC & is4);
    }

    o[nw - 1] &= lastMask;
  }

  std::free(zeroRow);
  current_.swap(next_);
}

Grid BitPackGameOfLife::getGrid() const { return current_.toGrid(); }

void *BitPackGameOfLife::getRowDataRaw(size_t row) {
  return current_.getRowData(row);
}

size_t BitPackGameOfLife::getStride() const { return wordsPerRow_; }

CellKind BitPackGameOfLife::getCellKind() const { return CellKind::BitPacked; }
