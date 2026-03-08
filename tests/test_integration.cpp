#include <gtest/gtest.h>

#include "gol_engine.h"
#include "grid.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

static std::string dataFile(const char *name) {
    return std::string(TEST_DATA_DIR) + "/" + name;
}

// Helper: extract all cell states from byte-based engine
static std::vector<bool> extractByteGrid(GameOfLife &engine, size_t rows,
                                         size_t cols) {
    std::vector<bool> cells(rows * cols);
    for (size_t r = 0; r < rows; r++) {
        auto *row = static_cast<uint8_t *>(engine.getRowDataRaw(r));
        for (size_t c = 0; c < cols; c++)
            cells[r * cols + c] = row[c];
    }
    return cells;
}

// Helper: extract all cell states from bitpack engine
static std::vector<bool> extractBitGrid(GameOfLife &engine, size_t rows,
                                        size_t cols) {
    std::vector<bool> cells(rows * cols);
    for (size_t r = 0; r < rows; r++) {
        auto *words = static_cast<uint64_t *>(engine.getRowDataRaw(r));
        for (size_t c = 0; c < cols; c++)
            cells[r * cols + c] = (words[c / 64] >> (c % 64)) & 1;
    }
    return cells;
}

// ── Engine equivalence: all CPU engines produce same result ─────────────────

TEST(Integration, EngineEquivalence) {
    const size_t rows = 64, cols = 64;
    const unsigned alive = 400;
    const int steps      = 50;

    // Simple
    std::mt19937 rng1(77);
    Grid g1(rows, cols, alive, rng1);
    SimpleGameOfLife simple(g1);
    for (int i = 0; i < steps; i++)
        simple.takeStep();
    auto simpleResult = extractByteGrid(simple, rows, cols);

    // SIMD
    std::mt19937 rng2(77);
    Grid g2(rows, cols, alive, rng2);
    SIMDGameOfLife simd(g2);
    for (int i = 0; i < steps; i++)
        simd.takeStep();
    auto simdResult = extractByteGrid(simd, rows, cols);

    ASSERT_EQ(simpleResult, simdResult) << "simple vs simd mismatch";

    // BitPack
    std::mt19937 rng3(77);
    BitGrid bg(rows, cols, alive, rng3);
    BitPackGameOfLife bitpack(bg);
    for (int i = 0; i < steps; i++)
        bitpack.takeStep();
    auto bitpackResult = extractBitGrid(bitpack, rows, cols);

    ASSERT_EQ(simpleResult, bitpackResult) << "simple vs bitpack mismatch";
}

// ── Oscillator period verification ──────────────────────────────────────────

TEST(Integration, Oscillators_Period138) {
    Grid initial(dataFile("p138_oscillators.txt"));
    size_t rows = initial.getNumRows(), cols = initial.getNumCols();

    // Snapshot initial state
    std::vector<bool> expected(rows * cols);
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            expected[r * cols + c] = initial.getCell(r, c);

    // Run 138 steps — oscillator should return to initial state
    Grid g(dataFile("p138_oscillators.txt"));
    SimpleGameOfLife engine(g);
    for (int i = 0; i < 138; i++)
        engine.takeStep();
    auto result = extractByteGrid(engine, rows, cols);
    ASSERT_EQ(result, expected);
}

// ── Glider displacement with SIMD engine ────────────────────────────────────

TEST(Integration, GliderDisplacement_SIMD) {
    Grid g(dataFile("glider.txt"));
    size_t rows = g.getNumRows(), cols = g.getNumCols();

    // Collect alive positions before
    std::vector<std::pair<size_t, size_t>> before;
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            if (g.getCell(r, c))
                before.emplace_back(r, c);

    SIMDGameOfLife engine(g);
    for (int i = 0; i < 4; i++)
        engine.takeStep();

    std::vector<std::pair<size_t, size_t>> after;
    for (size_t r = 0; r < rows; r++) {
        auto *row = static_cast<uint8_t *>(engine.getRowDataRaw(r));
        for (size_t c = 0; c < cols; c++)
            if (row[c])
                after.emplace_back(r, c);
    }

    ASSERT_EQ(after.size(), before.size());
    for (size_t i = 0; i < before.size(); i++) {
        EXPECT_EQ(after[i].first, before[i].first + 1);
        EXPECT_EQ(after[i].second, before[i].second + 1);
    }
}
