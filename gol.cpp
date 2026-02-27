#include "gol.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <omp.h>

#if __cpp_lib_simd
#  include <simd>
#  define GOL_USE_SIMD 1
   namespace { using std::simd_flag_default; using std::simd_select; }
#elif defined(__GNUC__) && !defined(__clang__) && __has_include(<experimental/simd>)
#  include <experimental/simd>
#  define GOL_USE_SIMD 1
   namespace std { using namespace std::experimental; } // pull native_simd etc. into std::
   namespace {
     constexpr auto simd_flag_default = std::experimental::element_aligned;
     template<class M, class V>
     V simd_select(M mask, V a, V b) {
       V r = b; std::experimental::where(mask, r) = a; return r;
     }
   }
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#elif defined(__AVX512BW__) || defined(__AVX2__)
#  include <immintrin.h>
#endif

// GOL rule: born if nb==3, survive if alive and nb==2.
static uint8_t golRule(uint8_t alive, uint8_t nb) {
  return (nb == 3) | (alive & (nb == 2));
}

// 8-neighbour count for a single cell with compile-time boundary knowledge.
// HasPrev/HasNext — a row above/below exists.
// HasLeft/HasRight — a column to the left/right exists.
template<bool HasPrev, bool HasNext, bool HasLeft, bool HasRight>
static uint8_t aliveNeighbours(const uint8_t *p, const uint8_t *c,
                                const uint8_t *n, size_t col) {
  uint8_t nb = 0;
  if constexpr (HasPrev) {
    if constexpr (HasLeft)  nb += p[col - 1];
    nb += p[col];
    if constexpr (HasRight) nb += p[col + 1];
  }
  if constexpr (HasLeft)  nb += c[col - 1];
  if constexpr (HasRight) nb += c[col + 1];
  if constexpr (HasNext) {
    if constexpr (HasLeft)  nb += n[col - 1];
    nb += n[col];
    if constexpr (HasRight) nb += n[col + 1];
  }
  return nb;
}

// ── Interior cell backends ────────────────────────────────────────────────────
// Processes interior columns [1, cols-1) for a fully-surrounded interior row.
// Backend chosen at compile time: std::simd → NEON → scalar auto-vectorised.
static void processInteriorCells(const uint8_t *__restrict__ p,
                                  const uint8_t *__restrict__ c,
                                  const uint8_t *__restrict__ n,
                                  uint8_t       *__restrict__ o, size_t cols) {
#if GOL_USE_SIMD
  // ── std::simd (C++26) / std::experimental::simd ────────────────────────
  // native_simd<uint8_t> maps to the widest native register (16 B on NEON,
  // 32 B on AVX2). Unaligned loads at ±1 are cheap on all modern CPUs.
  using Vec          = std::native_simd<uint8_t>;
  constexpr size_t W = Vec::size();
  size_t col = 1;
  for (; col + W <= cols - 1; col += W) {
    __builtin_prefetch(p + col + W * 4, 0, 0);
    __builtin_prefetch(c + col + W * 4, 0, 0);
    __builtin_prefetch(n + col + W * 4, 0, 0);
    const Vec pL(p + col - 1, simd_flag_default); // prev row, left
    const Vec pC(p + col,     simd_flag_default); // prev row, centre
    const Vec pR(p + col + 1, simd_flag_default); // prev row, right
    const Vec cL(c + col - 1, simd_flag_default); // curr row, left
    const Vec cC(c + col,     simd_flag_default); // curr row, centre (cell state)
    const Vec cR(c + col + 1, simd_flag_default); // curr row, right
    const Vec nL(n + col - 1, simd_flag_default); // next row, left
    const Vec nC(n + col,     simd_flag_default); // next row, centre
    const Vec nR(n + col + 1, simd_flag_default); // next row, right

    // 7 additions → 8-neighbour count for W cells simultaneously.
    const Vec nb = pL + pC + pR + cL + cR + nL + nC + nR;

    // simd_select(mask, a, b) picks a where mask is true, b elsewhere.
    const Vec result = simd_select(
        (nb == Vec(uint8_t(3))) | ((cC != Vec(uint8_t(0))) & (nb == Vec(uint8_t(2)))),
        Vec(uint8_t(1)), Vec(uint8_t(0)));

    result.copy_to(o + col, simd_flag_default);
  }
  for (; col < cols - 1; ++col) // tail: remaining cols after last full chunk
    o[col] = golRule(c[col], aliveNeighbours<true, true, true, true>(p, c, n, col));

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
  // ── ARM NEON intrinsics ───────────────────────────────────────────────
  // uint8x16_t   — 128-bit register holding 16 uint8_t lanes (W = 16)
  // vld1q_u8     — unaligned load of 16 bytes
  // vaddq_u8     — elementwise byte addition across 16 lanes
  // vceqq_u8     — compare equal: 0xFF per lane where equal, 0x00 elsewhere
  // vtstq_u8(a,a)— test non-zero: 0xFF where any bit is set, 0x00 elsewhere
  // vandq_u8     — bitwise AND across lanes
  // vorrq_u8     — bitwise OR across lanes
  // vdupq_n_u8   — broadcast one scalar byte to all 16 lanes
  // vst1q_u8     — store 16 bytes
  constexpr size_t W = 16;
  size_t col = 1;
  for (; col + W <= cols - 1; col += W) {
    __builtin_prefetch(p + col + W * 4, 0, 0);
    __builtin_prefetch(c + col + W * 4, 0, 0);
    __builtin_prefetch(n + col + W * 4, 0, 0);
    const uint8x16_t pL = vld1q_u8(p + col - 1);
    const uint8x16_t pC = vld1q_u8(p + col    );
    const uint8x16_t pR = vld1q_u8(p + col + 1);
    const uint8x16_t cL = vld1q_u8(c + col - 1);
    const uint8x16_t cC = vld1q_u8(c + col    ); // cell state for rule
    const uint8x16_t cR = vld1q_u8(c + col + 1);
    const uint8x16_t nL = vld1q_u8(n + col - 1);
    const uint8x16_t nC = vld1q_u8(n + col    );
    const uint8x16_t nR = vld1q_u8(n + col + 1);

    // 7 additions → 8-neighbour count for 16 cells simultaneously.
    uint8x16_t nb = vaddq_u8(pL, pC);
    nb = vaddq_u8(nb, pR);
    nb = vaddq_u8(nb, cL);
    nb = vaddq_u8(nb, cR);
    nb = vaddq_u8(nb, nL);
    nb = vaddq_u8(nb, nC);
    nb = vaddq_u8(nb, nR);

    // born    = 0xFF lanes where nb == 3
    // survive = 0xFF lanes where cC != 0  AND  nb == 2
    // result  = (born | survive) & 0x01   → normalise mask to 0/1 values
    const uint8x16_t born    = vceqq_u8(nb, vdupq_n_u8(3));
    const uint8x16_t survive = vandq_u8(vtstq_u8(cC, cC), vceqq_u8(nb, vdupq_n_u8(2)));
    const uint8x16_t result  = vandq_u8(vorrq_u8(born, survive), vdupq_n_u8(1));

    vst1q_u8(o + col, result);
  }
  for (; col < cols - 1; ++col) // tail: remaining cols after last full chunk
    o[col] = golRule(c[col], aliveNeighbours<true, true, true, true>(p, c, n, col));

#elif defined(__AVX512BW__)
  // ── AVX512BW intrinsics ───────────────────────────────────────────────
  // __m512i  — 512-bit register holding 64 uint8_t lanes (W = 64)
  // _mm512_loadu_si512       — unaligned load of 64 bytes (void* pointer)
  // _mm512_add_epi8          — elementwise byte addition across 64 lanes
  // _mm512_cmpeq_epi8_mask   — compare equal: __mmask64 (1 bit per lane)
  // _mm512_cmpgt_epi8_mask   — compare greater-than: __mmask64
  // _kand_mask64 / _kor_mask64 — bitwise AND/OR of two __mmask64 values
  // _mm512_maskz_set1_epi8   — broadcast 1 to lanes where mask bit is set, 0 elsewhere
  // _mm512_storeu_si512      — store 64 bytes (void* pointer)
  constexpr size_t W = 64;
  size_t col = 1;
  for (; col + W <= cols - 1; col += W) {
    __builtin_prefetch(p + col + W * 4, 0, 0);
    __builtin_prefetch(c + col + W * 4, 0, 0);
    __builtin_prefetch(n + col + W * 4, 0, 0);
    const __m512i pL = _mm512_loadu_si512(p + col - 1);
    const __m512i pC = _mm512_loadu_si512(p + col    );
    const __m512i pR = _mm512_loadu_si512(p + col + 1);
    const __m512i cL = _mm512_loadu_si512(c + col - 1);
    const __m512i cC = _mm512_loadu_si512(c + col    ); // cell state for rule
    const __m512i cR = _mm512_loadu_si512(c + col + 1);
    const __m512i nL = _mm512_loadu_si512(n + col - 1);
    const __m512i nC = _mm512_loadu_si512(n + col    );
    const __m512i nR = _mm512_loadu_si512(n + col + 1);

    // 7 additions → 8-neighbour count for 64 cells simultaneously.
    __m512i nb = _mm512_add_epi8(pL, pC);
    nb = _mm512_add_epi8(nb, pR);
    nb = _mm512_add_epi8(nb, cL);
    nb = _mm512_add_epi8(nb, cR);
    nb = _mm512_add_epi8(nb, nL);
    nb = _mm512_add_epi8(nb, nC);
    nb = _mm512_add_epi8(nb, nR);

    // born    = mask bits set where nb == 3
    // survive = mask bits set where cC > 0  AND  nb == 2
    // result  = 1 in lanes where (born | survive), 0 elsewhere
    const __mmask64 born    = _mm512_cmpeq_epi8_mask(nb, _mm512_set1_epi8(3));
    const __mmask64 survive = _kand_mask64(_mm512_cmpgt_epi8_mask(cC, _mm512_setzero_si512()),
                                           _mm512_cmpeq_epi8_mask(nb, _mm512_set1_epi8(2)));
    const __m512i result    = _mm512_maskz_set1_epi8(_kor_mask64(born, survive), 1);

    _mm512_storeu_si512(o + col, result);
  }
  for (; col < cols - 1; ++col) // tail: remaining cols after last full chunk
    o[col] = golRule(c[col], aliveNeighbours<true, true, true, true>(p, c, n, col));

#elif defined(__AVX2__)
  // ── AVX2 intrinsics ───────────────────────────────────────────────────
  // __m256i  — 256-bit register holding 32 uint8_t lanes (W = 32)
  // _mm256_loadu_si256    — unaligned load of 32 bytes (__m256i* pointer)
  // _mm256_add_epi8       — elementwise byte addition across 32 lanes
  // _mm256_cmpeq_epi8     — compare equal: 0xFF per lane where equal, 0x00 elsewhere
  // _mm256_cmpgt_epi8     — compare greater-than (signed): 0xFF where cC > 0
  // _mm256_and_si256      — bitwise AND across lanes
  // _mm256_or_si256       — bitwise OR across lanes
  // _mm256_storeu_si256   — store 32 bytes (__m256i* pointer)
  constexpr size_t W = 32;
  size_t col = 1;
  for (; col + W <= cols - 1; col += W) {
    __builtin_prefetch(p + col + W * 4, 0, 0);
    __builtin_prefetch(c + col + W * 4, 0, 0);
    __builtin_prefetch(n + col + W * 4, 0, 0);
    const __m256i pL = _mm256_loadu_si256((const __m256i *)(p + col - 1));
    const __m256i pC = _mm256_loadu_si256((const __m256i *)(p + col    ));
    const __m256i pR = _mm256_loadu_si256((const __m256i *)(p + col + 1));
    const __m256i cL = _mm256_loadu_si256((const __m256i *)(c + col - 1));
    const __m256i cC = _mm256_loadu_si256((const __m256i *)(c + col    )); // cell state for rule
    const __m256i cR = _mm256_loadu_si256((const __m256i *)(c + col + 1));
    const __m256i nL = _mm256_loadu_si256((const __m256i *)(n + col - 1));
    const __m256i nC = _mm256_loadu_si256((const __m256i *)(n + col    ));
    const __m256i nR = _mm256_loadu_si256((const __m256i *)(n + col + 1));

    // 7 additions → 8-neighbour count for 32 cells simultaneously.
    __m256i nb = _mm256_add_epi8(pL, pC);
    nb = _mm256_add_epi8(nb, pR);
    nb = _mm256_add_epi8(nb, cL);
    nb = _mm256_add_epi8(nb, cR);
    nb = _mm256_add_epi8(nb, nL);
    nb = _mm256_add_epi8(nb, nC);
    nb = _mm256_add_epi8(nb, nR);

    // born    = 0xFF lanes where nb == 3
    // survive = 0xFF lanes where cC > 0  AND  nb == 2
    // result  = (born | survive) & 0x01  → normalise mask to 0/1 values
    const __m256i born    = _mm256_cmpeq_epi8(nb, _mm256_set1_epi8(3));
    const __m256i survive = _mm256_and_si256(_mm256_cmpgt_epi8(cC, _mm256_setzero_si256()),
                                             _mm256_cmpeq_epi8(nb, _mm256_set1_epi8(2)));
    const __m256i result  = _mm256_and_si256(_mm256_or_si256(born, survive),
                                             _mm256_set1_epi8(1));

    _mm256_storeu_si256((__m256i *)(o + col), result);
  }
  for (; col < cols - 1; ++col) // tail: remaining cols after last full chunk
    o[col] = golRule(c[col], aliveNeighbours<true, true, true, true>(p, c, n, col));

#else
  // ── Scalar fallback ───────────────────────────────────────────────────
  // The compiler auto-vectorises with -O3 -march=native.
  for (size_t col = 1; col < cols - 1; ++col)
    o[col] = golRule(c[col], aliveNeighbours<true, true, true, true>(p, c, n, col));
#endif
}

// Full interior row: left/right border columns + interior cells.
static void processInteriorRow(const uint8_t *p, const uint8_t *c,
                                const uint8_t *n, uint8_t *o, size_t cols) {
  o[0]        = golRule(c[0],        aliveNeighbours<true, true, false, true >(p, c, n, 0));
  o[cols - 1] = golRule(c[cols - 1], aliveNeighbours<true, true, true,  false>(p, c, n, cols - 1));
  processInteriorCells(p, c, n, o, cols);
}

// Apply golRule + aliveNeighbours across a complete border row.
// HasPrev=false for the top row, HasNext=false for the bottom row.
// Splits into left corner | middle columns | right corner so each gets the
// correct HasLeft/HasRight instantiation.
template<bool HasPrev, bool HasNext>
static void processBorderRow(const uint8_t *p, const uint8_t *c,
                              const uint8_t *n, uint8_t *o, size_t cols) {
  o[0] = golRule(c[0], aliveNeighbours<HasPrev, HasNext, false, true>(p, c, n, 0));
  for (size_t col = 1; col + 1 < cols; ++col)
    o[col] = golRule(c[col], aliveNeighbours<HasPrev, HasNext, true, true>(p, c, n, col));
  o[cols - 1] = golRule(c[cols - 1],
                        aliveNeighbours<HasPrev, HasNext, true, false>(p, c, n, cols - 1));
}

// ── Grid implementation

Grid::Grid() : gridRows_(0), gridColumns_(0), cells_(nullptr) {}

Grid::Grid(size_t gridRows, size_t gridColumns)
    : gridRows_(gridRows), gridColumns_(gridColumns),
      cells_(new uint8_t[gridRows * gridColumns]()) {}

Grid::Grid(size_t gridRows, size_t gridColumns, unsigned int alive,
           std::mt19937 &rng)
    : gridRows_(gridRows), gridColumns_(gridColumns),
      cells_(new uint8_t[gridRows * gridColumns]()) {
  std::uniform_int_distribution<size_t> dist(0, gridRows * gridColumns - 1);
  size_t uniqueNumbers = 0;
  while (uniqueNumbers < alive) {
    size_t idx = dist(rng);
    if (!cells_[idx]) {
      cells_[idx] = true;
      uniqueNumbers++;
    }
  }
}

Grid::Grid(const std::string &filename)
    : gridRows_(0), gridColumns_(0), cells_(nullptr) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << "\n";
    return;
  }

  std::string line;
  std::vector<uint8_t> tempCells;
  bool firstLine = true;

  while (std::getline(file, line)) {
    size_t whiteSpace = std::count(line.begin(), line.end(), ' ');
    if (firstLine) {
      gridColumns_ = whiteSpace + 1;
      firstLine = false;
    }

    std::stringstream ss(line);
    std::string temp;
    while (ss >> temp) {
      tempCells.push_back(temp == "o");
    }
  }

  if (tempCells.empty())
    return;
  gridRows_ = tempCells.size() / gridColumns_;
  cells_ = new uint8_t[gridRows_ * gridColumns_];
  std::copy(tempCells.begin(), tempCells.end(), cells_);
}

Grid::Grid(Grid &&other) noexcept
    : gridRows_(other.gridRows_), gridColumns_(other.gridColumns_),
      cells_(other.cells_) {
  other.gridRows_ = 0;
  other.gridColumns_ = 0;
  other.cells_ = nullptr;
}

Grid &Grid::operator=(Grid &&other) noexcept {
  if (this != &other) {
    delete[] cells_;
    gridRows_ = other.gridRows_;
    gridColumns_ = other.gridColumns_;
    cells_ = other.cells_;
    other.gridRows_ = 0;
    other.gridColumns_ = 0;
    other.cells_ = nullptr;
  }
  return *this;
}

Grid::~Grid() { delete[] cells_; }

size_t Grid::getNumRows() const { return gridRows_; }
size_t Grid::getNumColumns() const { return gridColumns_; }

bool Grid::getCell(size_t row, size_t column) const {
  return cells_[row * gridColumns_ + column];
}

void Grid::setCell(size_t row, size_t column, bool cellStatus) {
  cells_[row * gridColumns_ + column] = cellStatus;
}

void Grid::swap(Grid &other) { std::swap(cells_, other.cells_); }

uint8_t *Grid::getRowPointer(size_t row) const {
  return cells_ + row * gridColumns_;
}

void Grid::setRow(size_t row, const uint8_t *rowCells) {
  std::copy(rowCells, rowCells + gridColumns_, cells_ + row * gridColumns_);
}

uint8_t *Grid::getCellsPointer() const { return cells_; }


size_t Grid::aliveCells() const {
  size_t count = 0;
  for (size_t i = 0; i < gridRows_ * gridColumns_; ++i) {
    if (cells_[i])
      count++;
  }
  return count;
}

void Grid::printGrid() const {
  for (size_t row = 0; row < gridRows_; ++row) {
    for (size_t col = 0; col < gridColumns_; ++col) {
      std::cout << (cells_[row * gridColumns_ + col] ? "o" : "-") << " ";
    }
    std::cout << "\n";
  }
}

void Grid::writeToFile(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot write to file " << filename << "\n";
    return;
  }

  for (size_t row = 0; row < gridRows_; ++row) {
    for (size_t col = 0; col < gridColumns_; ++col) {
      file << (cells_[row * gridColumns_ + col] ? "o" : "-");
      if (col < gridColumns_ - 1)
        file << " ";
    }
    file << "\n";
  }
}

// GameOfLife implementation
GameOfLife::GameOfLife(Grid &grid)
    : currentGrid_(std::move(grid)),
      newGrid_(currentGrid_.getNumRows(), currentGrid_.getNumColumns()),
      gridRows_(currentGrid_.getNumRows()),
      gridColumns_(currentGrid_.getNumColumns()),
      totalCells_(gridRows_ * gridColumns_) {}

void GameOfLife::takeStep() {
  const uint8_t *__restrict__ src = currentGrid_.getCellsPointer();
  uint8_t       *__restrict__ dst = newGrid_.getCellsPointer();
  const size_t rows = gridRows_;
  const size_t cols = gridColumns_;

#pragma omp parallel for schedule(static)
  for (size_t row = 0; row < rows; ++row) {
    const uint8_t *p = (row > 0)        ? src + (row - 1) * cols : nullptr;
    const uint8_t *c =                    src +  row      * cols;
    const uint8_t *n = (row < rows - 1) ? src + (row + 1) * cols : nullptr;
    uint8_t       *o =                    dst +  row      * cols;

    if      (row == 0)        processBorderRow<false, true >(p, c, n, o, cols);
    else if (row == rows - 1) processBorderRow<true,  false>(p, c, n, o, cols);
    else                      processInteriorRow(p, c, n, o, cols);
  }
  currentGrid_.swap(newGrid_);
}

const Grid &GameOfLife::getGrid() const { return currentGrid_; }

uint8_t *GameOfLife::getRowPointer(size_t row) {
  return currentGrid_.getRowPointer(row);
}

// ── Bit-parallel helpers ─────────────────────────────────────────────────────
// rowSum3: 3-input per-bit adder → 2-bit result (s1:s0) per bit position.
// For each bit k: (s1[k]<<1 | s0[k]) == popcount(L[k], C[k], R[k]).
static inline void rowSum3(uint64_t L, uint64_t C, uint64_t R,
                            uint64_t &s1, uint64_t &s0) {
  uint64_t t = L ^ C;
  s0 = t ^ R;
  s1 = (L & C) | (t & R);
}

// sum9: add three 2-bit row-sums to a 4-bit result per bit position.
// Inputs: three (hi:lo) pairs, each a row's 3-column popcount (values 0–3).
// Output: o3:o2:o1:o0 holds the 9-cell neighbourhood total (values 0–9),
//         with the cell itself included via the centre column of its own row.
static inline void sum9(uint64_t p1, uint64_t p0,
                        uint64_t c1, uint64_t c0,
                        uint64_t n1, uint64_t n0,
                        uint64_t &o3, uint64_t &o2,
                        uint64_t &o1, uint64_t &o0) {
  // Stage 1: add prev-row sum + curr-row sum (2-bit + 2-bit → 3-bit)
  uint64_t t0    = p0 ^ c0;
  uint64_t carry = p0 & c0;
  uint64_t q     = p1 ^ c1;
  uint64_t t1    = q ^ carry;
  uint64_t t2    = (p1 & c1) | (q & carry);  // t2:t1:t0 in [0,6]

  // Stage 2: add 3-bit running total + next-row sum (2-bit, max 3)
  o0    = t0 ^ n0;
  carry = t0 & n0;
  uint64_t q1 = t1 ^ n1;
  o1    = q1 ^ carry;
  carry = (t1 & n1) | (q1 & carry);
  o2    = t2 ^ carry;
  o3    = t2 & carry;
}

// ── BitGrid implementation ────────────────────────────────────────────────────
BitGrid::BitGrid() : rows_(0), cols_(0), wordsPerRow_(0), words_(nullptr) {}

BitGrid::BitGrid(size_t rows, size_t cols)
    : rows_(rows), cols_(cols),
      wordsPerRow_((cols + 63) / 64),
      words_(new uint64_t[rows * ((cols + 63) / 64)]()) {}

BitGrid::BitGrid(const Grid &g)
    : rows_(g.getNumRows()), cols_(g.getNumColumns()),
      wordsPerRow_((g.getNumColumns() + 63) / 64),
      words_(new uint64_t[g.getNumRows() * ((g.getNumColumns() + 63) / 64)]()) {
  for (size_t r = 0; r < rows_; ++r)
    for (size_t c = 0; c < cols_; ++c)
      if (g.getCell(r, c))
        words_[r * wordsPerRow_ + c / 64] |= uint64_t(1) << (c % 64);
}

Grid BitGrid::toGrid() const {
  Grid g(rows_, cols_);
  for (size_t r = 0; r < rows_; ++r)
    for (size_t c = 0; c < cols_; ++c)
      g.setCell(r, c, (words_[r * wordsPerRow_ + c / 64] >> (c % 64)) & 1);
  return g;
}

BitGrid::BitGrid(BitGrid &&other) noexcept
    : rows_(other.rows_), cols_(other.cols_),
      wordsPerRow_(other.wordsPerRow_), words_(other.words_) {
  other.rows_ = other.cols_ = other.wordsPerRow_ = 0;
  other.words_ = nullptr;
}

BitGrid &BitGrid::operator=(BitGrid &&other) noexcept {
  if (this != &other) {
    delete[] words_;
    rows_ = other.rows_; cols_ = other.cols_;
    wordsPerRow_ = other.wordsPerRow_; words_ = other.words_;
    other.rows_ = other.cols_ = other.wordsPerRow_ = 0;
    other.words_ = nullptr;
  }
  return *this;
}

BitGrid::~BitGrid() { delete[] words_; }

void BitGrid::swap(BitGrid &other) { std::swap(words_, other.words_); }

size_t BitGrid::aliveCells() const {
  size_t count = 0;
  for (size_t i = 0; i < rows_ * wordsPerRow_; ++i)
    count += __builtin_popcountll(words_[i]);
  return count;
}

// ── NEON helper ───────────────────────────────────────────────────────────────
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
// Bitwise NOT for uint64x2_t via byte-level mvn (correct for bitwise NOT).
static inline uint64x2_t vnot64(uint64x2_t x) {
  return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}
#endif // __ARM_NEON__

// ── BitGameOfLife implementation ──────────────────────────────────────────────
BitGameOfLife::BitGameOfLife(Grid &grid)
    : current_(grid),
      next_(grid.getNumRows(), grid.getNumColumns()),
      rows_(grid.getNumRows()),
      cols_(grid.getNumColumns()),
      wordsPerRow_((grid.getNumColumns() + 63) / 64) {}

void BitGameOfLife::takeStep() {
  const size_t    nw  = wordsPerRow_;
  const uint64_t *src = current_.getWords();
  uint64_t       *dst = next_.getWords();

  // Padding mask: top-bit of the last word beyond cols_ must stay 0.
  const uint64_t lastMask = (cols_ % 64 == 0) ? ~uint64_t(0)
                                               : (uint64_t(1) << (cols_ % 64)) - 1;

  // Zero row: used in place of nullptr for first/last row to eliminate all
  // branches from the inner loops.
  uint64_t *zeroRow = static_cast<uint64_t *>(std::calloc(nw, sizeof(uint64_t)));

#pragma omp parallel for schedule(static)
  for (size_t row = 0; row < rows_; ++row) {
    const uint64_t *p = (row > 0)         ? src + (row - 1) * nw : zeroRow;
    const uint64_t *c =                     src +  row      * nw;
    const uint64_t *n = (row < rows_ - 1) ? src + (row + 1) * nw : zeroRow;
    uint64_t       *o =                     dst +  row      * nw;

    // Convention: bit k of word w = cell at column (64*w + k).
    // Left-neighbour plane  = row shifted <<1 with carry from word[w-1] >> 63.
    // Right-neighbour plane = row shifted >>1 with carry from word[w+1] & 1.
    // Self (cC) is included in the curr-row column sum, so the 9-cell total
    // ranges 0-9 and the rule becomes: alive if sum==3 || (was_alive && sum==4).
     
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

    // Clear padding bits in the last word to preserve the zero-pad invariant.
    o[nw - 1] &= lastMask;
  }

  std::free(zeroRow);
  current_.swap(next_);
}

Grid BitGameOfLife::getGrid() const { return current_.toGrid(); }
