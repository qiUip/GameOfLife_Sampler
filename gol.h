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

#endif // GOL_H
