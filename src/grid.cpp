#include "grid.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// -- Template helpers for Grid and BitGrid -----------------------------------

struct FileData {
    size_t rows = 0, cols = 0;
    std::vector<bool> cells;
};

static FileData parseGridFile(const std::string &filename) {
    FileData fd;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return fd;
    }

    std::string line;
    bool firstLine = true;
    while (std::getline(file, line)) {
        if (firstLine) {
            fd.cols   = std::count(line.begin(), line.end(), ' ') + 1;
            firstLine = false;
        }
        std::stringstream ss(line);
        std::string token;
        while (ss >> token)
            fd.cells.push_back(token == "o");
    }

    if (!fd.cells.empty())
        fd.rows = fd.cells.size() / fd.cols;
    return fd;
}

template <typename GridType>
static void fillFromFile(GridType &grid, const FileData &fd) {
    for (size_t r = 0; r < fd.rows; r++)
        for (size_t c = 0; c < fd.cols; c++)
            if (fd.cells[r * fd.cols + c])
                grid.setCell(r, c, true);
}

template <typename GridType> static void printGridImpl(const GridType &grid) {
    for (size_t row = 0; row < grid.getNumRows(); ++row) {
        for (size_t col = 0; col < grid.getNumCols(); ++col)
            std::cout << (grid.getCell(row, col) ? "o" : "-") << " ";
        std::cout << "\n";
    }
}

template <typename GridType>
static void writeToFileImpl(const GridType &grid, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write to file " << filename << "\n";
        return;
    }
    for (size_t row = 0; row < grid.getNumRows(); ++row) {
        for (size_t col = 0; col < grid.getNumCols(); ++col) {
            file << (grid.getCell(row, col) ? "o" : "-");
            if (col < grid.getNumCols() - 1)
                file << " ";
        }
        file << "\n";
    }
}

// -- Grid implementation -----------------------------------------------------

Grid::Grid(size_t rows, size_t cols) : GridStorage(rows, cols, cols) {
}

Grid::Grid(size_t rows, size_t cols, unsigned int alive, std::mt19937 &rng)
    : Grid(rows, cols) {
    std::uniform_int_distribution<size_t> dist(0, rows * cols - 1);
    size_t placed = 0;
    while (placed < alive) {
        size_t idx = dist(rng);
        if (!data_[idx]) {
            data_[idx] = true;
            placed++;
        }
    }
}

Grid::Grid(const std::string &filename) : GridStorage() {
    auto fd = parseGridFile(filename);
    if (fd.cells.empty())
        return;
    *this = Grid(fd.rows, fd.cols);
    fillFromFile(*this, fd);
}

bool Grid::getCell(size_t row, size_t column) const {
    return data_[row * cols_ + column];
}

void Grid::setCell(size_t row, size_t column, bool cellStatus) {
    data_[row * cols_ + column] = cellStatus;
}

size_t Grid::aliveCells() const {
    size_t count = 0;
    for (size_t i = 0; i < rows_ * cols_; ++i)
        if (data_[i])
            count++;
    return count;
}

void Grid::printGrid() const {
    printGridImpl(*this);
}

void Grid::writeToFile(const std::string &filename) const {
    writeToFileImpl(*this, filename);
}

// -- BitGrid implementation --------------------------------------------------

BitGrid::BitGrid(size_t rows, size_t cols)
    : GridStorage(rows, cols, (cols + 63) / 64),
      wordsPerRow_((cols + 63) / 64) {
}

BitGrid::BitGrid(size_t rows, size_t cols, unsigned int alive,
                 std::mt19937 &rng)
    : BitGrid(rows, cols) {
    std::uniform_int_distribution<size_t> dist(0, rows * cols - 1);
    size_t placed = 0;
    while (placed < alive) {
        size_t idx = dist(rng);
        size_t r = idx / cols, c = idx % cols;
        if (!getCell(r, c)) {
            setCell(r, c, true);
            placed++;
        }
    }
}

BitGrid::BitGrid(const std::string &filename) : GridStorage() {
    auto fd = parseGridFile(filename);
    if (fd.cells.empty())
        return;
    *this = BitGrid(fd.rows, fd.cols);
    fillFromFile(*this, fd);
}

size_t BitGrid::aliveCells() const {
    size_t count = 0;
    for (size_t i = 0; i < rows_ * wordsPerRow_; ++i)
        count += __builtin_popcountll(data_[i]);
    return count;
}

void BitGrid::printGrid() const {
    printGridImpl(*this);
}

void BitGrid::writeToFile(const std::string &filename) const {
    writeToFileImpl(*this, filename);
}
