#include "gol.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <omp.h>

// Grid implementation

Grid::Grid() : gridRows_(0), gridColumns_(0), cells_(nullptr) {}

Grid::Grid(size_t gridRows, size_t gridColumns)
    : gridRows_(gridRows), gridColumns_(gridColumns),
      cells_(new bool[gridRows * gridColumns]()) {}

Grid::Grid(size_t gridRows, size_t gridColumns, unsigned int alive,
           std::mt19937 &rng)
    : gridRows_(gridRows), gridColumns_(gridColumns),
      cells_(new bool[gridRows * gridColumns]()) {
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
  std::vector<bool> tempCells;
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
  cells_ = new bool[gridRows_ * gridColumns_];
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

bool *Grid::getRowPointer(size_t row) const {
  return cells_ + row * gridColumns_;
}

void Grid::setRow(size_t row, const bool *rowCells) {
  std::copy(rowCells, rowCells + gridColumns_, cells_ + row * gridColumns_);
}

bool *Grid::getCellsPointer() const { return cells_; }

size_t Grid::aliveNeighbours(size_t row, size_t column) const {
  constexpr int8_t offsets[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                    {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  size_t count = 0;
  const size_t maxRow = gridRows_ - 1;
  const size_t maxCol = gridColumns_ - 1;

  for (const auto &offset : offsets) {
    const size_t neighbourRow = row + offset[0];
    const size_t neighbourCol = column + offset[1];
    if (neighbourRow <= maxRow && neighbourCol <= maxCol) {
      count += cells_[neighbourRow * gridColumns_ + neighbourCol];
    }
  }
  return count;
}

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
#pragma omp parallel for
  for (size_t index = 0; index < totalCells_; index++) {

    const size_t row = index / gridColumns_;
    const size_t col = index % gridColumns_;

    const size_t aliveNeighbours = currentGrid_.aliveNeighbours(row, col);
    bool alive = currentGrid_.getCell(row, col);

    if (!alive && aliveNeighbours == 3) {
      alive = true;
    } else if (alive && (aliveNeighbours < 2 || aliveNeighbours > 3)) {
      alive = false;
    }
    newGrid_.setCell(row, col, alive);
  }
  currentGrid_.swap(newGrid_);
}

const Grid &GameOfLife::getGrid() const { return currentGrid_; }

bool *GameOfLife::getRowPointer(size_t row) {
  return currentGrid_.getRowPointer(row);
}
