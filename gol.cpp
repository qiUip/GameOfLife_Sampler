#include "gol.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// ── Grid implementation ──────────────────────────────────────────────────────

Grid::Grid(size_t rows, size_t cols) : GridStorage(rows, cols, cols) {}

Grid::Grid(size_t rows, size_t cols, unsigned int alive, std::mt19937 &rng)
    : GridStorage(rows, cols, cols) {
  std::uniform_int_distribution<size_t> dist(0, rows * cols - 1);
  size_t uniqueNumbers = 0;
  while (uniqueNumbers < alive) {
    size_t idx = dist(rng);
    if (!data_[idx]) {
      data_[idx] = true;
      uniqueNumbers++;
    }
  }
}

Grid::Grid(const std::string &filename) : GridStorage() {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << "\n";
    return;
  }

  std::string line;
  std::vector<CellType> tempCells;
  bool firstLine = true;

  while (std::getline(file, line)) {
    size_t whiteSpace = std::count(line.begin(), line.end(), ' ');
    if (firstLine) {
      cols_ = whiteSpace + 1;
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
  rows_ = tempCells.size() / cols_;
  data_ = new CellType[rows_ * cols_];
  std::copy(tempCells.begin(), tempCells.end(), data_);
}

bool Grid::getCell(size_t row, size_t column) const {
  return data_[row * cols_ + column];
}

void Grid::setCell(size_t row, size_t column, bool cellStatus) {
  data_[row * cols_ + column] = cellStatus;
}

size_t Grid::aliveCells() const {
  size_t count = 0;
  for (size_t i = 0; i < rows_ * cols_; ++i) {
    if (data_[i])
      count++;
  }
  return count;
}

void Grid::printGrid() const {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::cout << (data_[row * cols_ + col] ? "o" : "-") << " ";
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

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      file << (data_[row * cols_ + col] ? "o" : "-");
      if (col < cols_ - 1)
        file << " ";
    }
    file << "\n";
  }
}
