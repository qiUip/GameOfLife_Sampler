#ifndef GRID_H
#define GRID_H

#include <cstdint>
#include <cstring>
#include <random>
#include <string>

// ── Common grid storage ─────────────────────────────────────────────────────

template <typename T> class GridStorage {
public:
    using CellType = T;

    GridStorage() : rows_(0), cols_(0), data_(nullptr) {
    }

    GridStorage(size_t rows, size_t cols, size_t stride)
        : rows_(rows), cols_(cols), data_(new T[rows * stride]()) {
    }

    GridStorage(GridStorage &&other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
        other.rows_ = other.cols_ = 0;
        other.data_               = nullptr;
    }

    GridStorage &operator=(GridStorage &&other) noexcept {
        if (this != &other) {
            delete[] data_;
            rows_       = other.rows_;
            cols_       = other.cols_;
            data_       = other.data_;
            other.rows_ = other.cols_ = 0;
            other.data_               = nullptr;
        }
        return *this;
    }

    ~GridStorage() {
        delete[] data_;
    }

    size_t getNumRows() const {
        return rows_;
    }
    size_t getNumCols() const {
        return cols_;
    }

    T *getData() {
        return data_;
    }
    const T *getData() const {
        return data_;
    }

    void swap(GridStorage &other) {
        T *tmp      = data_;
        data_       = other.data_;
        other.data_ = tmp;
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

    size_t getStride() const {
        return cols_;
    }

    CellType *getRowData(size_t row) {
        return data_ + row * cols_;
    }
    const CellType *getRowData(size_t row) const {
        return data_ + row * cols_;
    }

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
    BitGrid(size_t rows, size_t cols, unsigned int alive, std::mt19937 &rng);
    BitGrid(const std::string &filename);

    size_t getStride() const {
        return wordsPerRow_;
    }

    CellType *getRowData(size_t row) {
        return data_ + row * wordsPerRow_;
    }
    const CellType *getRowData(size_t row) const {
        return data_ + row * wordsPerRow_;
    }

    void setRow(size_t row, const CellType *src) {
        std::memcpy(data_ + row * wordsPerRow_, src,
                    wordsPerRow_ * sizeof(CellType));
    }

    bool getCell(size_t row, size_t col) const {
        return (data_[row * wordsPerRow_ + col / 64] >> (col % 64)) & 1;
    }

    void setCell(size_t row, size_t col, bool val) {
        uint64_t &word = data_[row * wordsPerRow_ + col / 64];
        uint64_t bit   = uint64_t(1) << (col % 64);
        if (val)
            word |= bit;
        else
            word &= ~bit;
    }

    size_t aliveCells() const;
    void printGrid() const;
    void writeToFile(const std::string &filename) const;

private:
    size_t wordsPerRow_ = 0;
};

#endif // GRID_H
