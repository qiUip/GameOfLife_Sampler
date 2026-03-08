#ifndef GOL_ENGINE_H
#define GOL_ENGINE_H

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

    // Static helpers exposed for unit testing.
    static uint8_t golRule(uint8_t alive, uint8_t nb);

    template <bool HasPrev, bool HasNext, bool HasLeft, bool HasRight>
    static uint8_t aliveNeighbours(const uint8_t *p, const uint8_t *c,
                                   const uint8_t *n, size_t col);

    static void processInteriorCells(const uint8_t *p, const uint8_t *c,
                                     const uint8_t *n, uint8_t *o, size_t cols);

    static void processInteriorRow(const uint8_t *p, const uint8_t *c,
                                   const uint8_t *n, uint8_t *o, size_t cols);

    template <bool HasPrev, bool HasNext>
    static void processBorderRow(const uint8_t *p, const uint8_t *c,
                                 const uint8_t *n, uint8_t *o, size_t cols);

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

    // Static helpers exposed for unit testing.
    static void rowSum3(uint64_t L, uint64_t C, uint64_t R, uint64_t &s1,
                        uint64_t &s0);

    static void sum9(uint64_t p1, uint64_t p0, uint64_t c1, uint64_t c0,
                     uint64_t n1, uint64_t n0, uint64_t &o3, uint64_t &o2,
                     uint64_t &o1, uint64_t &o0);

private:
    BitGrid current_, next_;
    size_t wordsPerRow_;
};

#endif // GOL_ENGINE_H
