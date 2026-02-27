#ifndef GOL_H
#define GOL_H

#include <cstdint>
#include <random>
#include <string>

class Grid {
public:
  Grid();
  Grid(size_t gridRows, size_t gridColumns);
  Grid(size_t gridRows, size_t gridColumns, unsigned int alive,
       std::mt19937 &rng);
  Grid(const std::string &filename);

  Grid(Grid &&other) noexcept;
  Grid &operator=(Grid &&other) noexcept;
  ~Grid();

  size_t getNumRows() const;
  size_t getNumColumns() const;
  bool getCell(size_t row, size_t column) const;
  void setCell(size_t row, size_t column, bool cellStatus);
  void swap(Grid &other);
  uint8_t *getRowPointer(size_t row) const;
  void setRow(size_t row, const uint8_t *rowCells);
  uint8_t *getCellsPointer() const;
  size_t aliveCells() const;
  void printGrid() const;
  void writeToFile(const std::string &filename) const;

private:
  size_t gridRows_, gridColumns_;
  uint8_t *cells_;
};

class GameOfLife {
public:
  explicit GameOfLife(Grid &grid);
  void takeStep();
  const Grid &getGrid() const;
  uint8_t *getRowPointer(size_t row);

private:
  Grid currentGrid_;
  Grid newGrid_;
  size_t gridRows_;
  size_t gridColumns_;
  size_t totalCells_;
};

// ── Bit-packed grid: 1 bit per cell, 64 cells per uint64_t word ─────────────
class BitGrid {
public:
  BitGrid();
  BitGrid(size_t rows, size_t cols);
  explicit BitGrid(const Grid &g);   // convert from byte grid
  Grid toGrid() const;               // convert back to byte grid

  BitGrid(BitGrid &&other) noexcept;
  BitGrid &operator=(BitGrid &&other) noexcept;
  ~BitGrid();

  size_t getNumRows() const { return rows_; }
  size_t getNumCols() const { return cols_; }
  size_t getWordsPerRow() const { return wordsPerRow_; }

  uint64_t       *getWords()       { return words_; }
  const uint64_t *getWords() const { return words_; }

  void   swap(BitGrid &other);
  size_t aliveCells() const;

private:
  size_t   rows_, cols_, wordsPerRow_;
  uint64_t *words_;
};

// ── Bit-parallel Game of Life ────────────────────────────────────────────────
class BitGameOfLife {
public:
  explicit BitGameOfLife(Grid &grid);
  void takeStep();
  Grid getGrid() const;   // converts BitGrid → Grid on demand

private:
  BitGrid current_, next_;
  size_t  rows_, cols_, wordsPerRow_;
};

#endif // GOL_H
