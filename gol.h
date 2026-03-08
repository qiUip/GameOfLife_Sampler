#ifndef GOL_H
#define GOL_H

#include "grid.h"
#include <string>

// ── Virtual base class for Game of Life engines ──────────────────────────────

enum class CellKind { Byte, BitPacked };

class GameOfLife {
public:
    virtual ~GameOfLife()   = default;
    virtual void takeStep() = 0;

    size_t getNumRows() const {
        return rows_;
    }
    size_t getNumCols() const {
        return cols_;
    }

    virtual void *getRowDataRaw(size_t row) = 0;
    virtual size_t getStride() const        = 0;
    virtual CellKind getCellKind() const    = 0;
    virtual void commitBoundaries() {
    }
    virtual void sync() {
    }
    virtual void printGrid() const {
    }
    virtual void writeToFile(const std::string &filename) const {
        (void)filename;
    }

protected:
    size_t rows_, cols_;
};

// ── Simple engine (basic OpenMP, no SIMD) ────────────────────────────────────

class SimpleGameOfLife : public GameOfLife {
public:
    explicit SimpleGameOfLife(Grid &grid);
    void takeStep() override;
    void *getRowDataRaw(size_t row) override;
    size_t getStride() const override;
    CellKind getCellKind() const override;
    void printGrid() const override {
        currentGrid_.printGrid();
    }
    void writeToFile(const std::string &f) const override {
        currentGrid_.writeToFile(f);
    }

private:
    Grid currentGrid_;
    Grid newGrid_;
};

// ── Byte/SIMD engine ─────────────────────────────────────────────────────────

class SIMDGameOfLife : public GameOfLife {
public:
    explicit SIMDGameOfLife(Grid &grid);
    void takeStep() override;
    void *getRowDataRaw(size_t row) override;
    size_t getStride() const override;
    CellKind getCellKind() const override;
    void printGrid() const override {
        currentGrid_.printGrid();
    }
    void writeToFile(const std::string &f) const override {
        currentGrid_.writeToFile(f);
    }

private:
    Grid currentGrid_;
    Grid newGrid_;
};

// ── Bit-packed engine ────────────────────────────────────────────────────────

class BitPackGameOfLife : public GameOfLife {
public:
    explicit BitPackGameOfLife(BitGrid &grid);
    void takeStep() override;
    void *getRowDataRaw(size_t row) override;
    size_t getStride() const override;
    CellKind getCellKind() const override;
    void printGrid() const override {
        current_.printGrid();
    }
    void writeToFile(const std::string &f) const override {
        current_.writeToFile(f);
    }

private:
    BitGrid current_, next_;
    size_t wordsPerRow_;
};

#endif // GOL_H
