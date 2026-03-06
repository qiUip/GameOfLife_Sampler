#ifndef GOL_H
#define GOL_H

#include <cstdint>
#include <cstring>
#include <random>
#include <string>

// ── Common grid storage ─────────────────────────────────────────────────────

template<typename T>
class GridStorage {
public:
  using CellType = T;

  GridStorage() : rows_(0), cols_(0), data_(nullptr) {}

  GridStorage(size_t rows, size_t cols, size_t stride)
      : rows_(rows), cols_(cols),
        data_(new T[rows * stride]()) {}

  GridStorage(GridStorage &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.rows_ = other.cols_ = 0;
    other.data_ = nullptr;
  }

  GridStorage &operator=(GridStorage &&other) noexcept {
    if (this != &other) {
      delete[] data_;
      rows_ = other.rows_; cols_ = other.cols_;
      data_ = other.data_;
      other.rows_ = other.cols_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  ~GridStorage() { delete[] data_; }

  size_t getNumRows() const { return rows_; }
  size_t getNumCols() const { return cols_; }

  T       *getData()       { return data_; }
  const T *getData() const { return data_; }

  void swap(GridStorage &other) {
    T *tmp = data_; data_ = other.data_; other.data_ = tmp;
  }

protected:
  size_t rows_, cols_;
  T *data_;
};

// ── Byte grid (1 byte per cell) ─────────────────────────────────────────────

class Grid : public GridStorage<uint8_t> {
public:
  Grid() = default;
  Grid(size_t rows, size_t cols);
  Grid(size_t rows, size_t cols, unsigned int alive, std::mt19937 &rng);
  Grid(const std::string &filename);

  size_t getStride() const { return cols_; }

  CellType       *getRowData(size_t row)       { return data_ + row * cols_; }
  const CellType *getRowData(size_t row) const { return data_ + row * cols_; }

  void setRow(size_t row, const CellType *src) {
    std::memcpy(data_ + row * cols_, src, cols_ * sizeof(CellType));
  }

  bool getCell(size_t row, size_t column) const;
  void setCell(size_t row, size_t column, bool cellStatus);
  size_t aliveCells() const;
  void printGrid() const;
  void writeToFile(const std::string &filename) const;
};

// ── Bit-packed grid (1 bit per cell, 64 cells per uint64_t) ─────────────────

class BitGrid : public GridStorage<uint64_t> {
public:
  BitGrid() = default;
  BitGrid(size_t rows, size_t cols);
  explicit BitGrid(const Grid &g);
  Grid toGrid() const;

  size_t getStride() const { return wordsPerRow_; }

  CellType       *getRowData(size_t row)       { return data_ + row * wordsPerRow_; }
  const CellType *getRowData(size_t row) const { return data_ + row * wordsPerRow_; }

  void setRow(size_t row, const CellType *src) {
    std::memcpy(data_ + row * wordsPerRow_, src, wordsPerRow_ * sizeof(CellType));
  }

  size_t aliveCells() const;

private:
  size_t wordsPerRow_ = 0;
};

// ── Virtual base class for Game of Life engines ──────────────────────────────

enum class CellKind { Byte, BitPacked };

class GameOfLife{
public:
  virtual ~GameOfLife() = default;
  virtual void takeStep() = 0;
  virtual Grid getGrid() const = 0;

  size_t getNumRows() const { return rows_; }
  size_t getNumCols() const { return cols_; }

  virtual void  *getRowDataRaw(size_t row) = 0;
  virtual size_t getStride() const = 0;
  virtual CellKind getCellKind() const = 0;
  virtual void commitBoundaries() {}

protected:
  size_t rows_, cols_;
};

// ── Byte/SIMD engine ─────────────────────────────────────────────────────────

class SIMDGameOfLife : public GameOfLife{
public:
  explicit SIMDGameOfLife(Grid &grid);
  void takeStep() override;
  Grid getGrid() const override;
  void  *getRowDataRaw(size_t row) override;
  size_t getStride() const override;
  CellKind getCellKind() const override;

private:
  Grid currentGrid_;
  Grid newGrid_;
};

// ── Bit-packed engine ────────────────────────────────────────────────────────

class BitPackGameOfLife : public GameOfLife{
public:
  explicit BitPackGameOfLife(Grid &grid);
  explicit BitPackGameOfLife(BitGrid &grid);
  void takeStep() override;
  Grid getGrid() const override;
  void  *getRowDataRaw(size_t row) override;
  size_t getStride() const override;
  CellKind getCellKind() const override;

private:
  BitGrid current_, next_;
  size_t wordsPerRow_;
};

// ── CUDA engines ────────────────────────────────────────────────────────────

#if GOL_CUDA

class CUDAGameOfLife : public GameOfLife {
public:
  explicit CUDAGameOfLife(Grid &grid);
  ~CUDAGameOfLife();
  void takeStep() override;
  Grid getGrid() const override;
  void  *getRowDataRaw(size_t row) override;
  size_t getStride() const override;
  CellKind getCellKind() const override;
  void commitBoundaries() override;

private:
  uint8_t *d_current_, *d_next_;
  Grid hostGrid_;
};

class CUDABitPackGameOfLife : public GameOfLife {
public:
  explicit CUDABitPackGameOfLife(Grid &grid);
  explicit CUDABitPackGameOfLife(BitGrid &grid);
  ~CUDABitPackGameOfLife();
  void takeStep() override;
  Grid getGrid() const override;
  void  *getRowDataRaw(size_t row) override;
  size_t getStride() const override;
  CellKind getCellKind() const override;
  void commitBoundaries() override;

private:
  uint64_t *d_current_, *d_next_;
  BitGrid hostGrid_;
  size_t wordsPerRow_;
};

#endif // GOL_CUDA

#endif // GOL_H
