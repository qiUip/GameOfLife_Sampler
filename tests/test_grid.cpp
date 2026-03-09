#include <gtest/gtest.h>

#include "grid.h"

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

static std::string dataFile(const char *name) {
    return std::string(TEST_DATA_DIR) + "/" + name;
}

// -- Grid: setCell / getCell -------------------------------------------------

TEST(Grid, SetGetCell) {
    Grid g(6, 8);
    // All cells start dead
    for (size_t r = 0; r < 6; r++)
        for (size_t c = 0; c < 8; c++)
            EXPECT_FALSE(g.getCell(r, c));

    // Interior cell
    g.setCell(2, 3, true);
    EXPECT_TRUE(g.getCell(2, 3));
    EXPECT_FALSE(g.getCell(2, 2));
    EXPECT_FALSE(g.getCell(3, 3));

    // Corner cells
    g.setCell(0, 0, true);
    g.setCell(0, 7, true);
    g.setCell(5, 0, true);
    g.setCell(5, 7, true);
    EXPECT_TRUE(g.getCell(0, 0));
    EXPECT_TRUE(g.getCell(0, 7));
    EXPECT_TRUE(g.getCell(5, 0));
    EXPECT_TRUE(g.getCell(5, 7));

    // Setting alive on already-alive cell is idempotent
    g.setCell(2, 3, true);
    EXPECT_TRUE(g.getCell(2, 3));
    EXPECT_EQ(g.aliveCells(), 5u);
}

TEST(Grid, SetCellClear) {
    Grid g(4, 4);
    g.setCell(1, 1, true);
    EXPECT_TRUE(g.getCell(1, 1));
    g.setCell(1, 1, false);
    EXPECT_FALSE(g.getCell(1, 1));

    // Setting false on already-dead cell is idempotent
    g.setCell(0, 0, false);
    EXPECT_FALSE(g.getCell(0, 0));

    // Corner clear
    g.setCell(3, 3, true);
    EXPECT_TRUE(g.getCell(3, 3));
    g.setCell(3, 3, false);
    EXPECT_FALSE(g.getCell(3, 3));
}

TEST(Grid, AliveCellsCounting) {
    Grid g(4, 4);
    EXPECT_EQ(g.aliveCells(), 0u);

    g.setCell(0, 0, true);
    EXPECT_EQ(g.aliveCells(), 1u);

    g.setCell(3, 3, true);
    g.setCell(1, 2, true);
    EXPECT_EQ(g.aliveCells(), 3u);

    g.setCell(0, 0, false);
    EXPECT_EQ(g.aliveCells(), 2u);
}

TEST(Grid, SingleRowGrid) {
    Grid g(1, 10);
    g.setCell(0, 5, true);
    EXPECT_TRUE(g.getCell(0, 5));
    EXPECT_EQ(g.aliveCells(), 1u);
}

TEST(Grid, SingleColumnGrid) {
    Grid g(10, 1);
    g.setCell(7, 0, true);
    EXPECT_TRUE(g.getCell(7, 0));
    EXPECT_EQ(g.aliveCells(), 1u);
}

// -- BitGrid: setCell / getCell ----------------------------------------------

TEST(BitGrid, SetGetCell) {
    BitGrid bg(6, 8);
    for (size_t r = 0; r < 6; r++)
        for (size_t c = 0; c < 8; c++)
            EXPECT_FALSE(bg.getCell(r, c));

    // Interior cell
    bg.setCell(2, 3, true);
    EXPECT_TRUE(bg.getCell(2, 3));
    EXPECT_FALSE(bg.getCell(2, 2));
    EXPECT_FALSE(bg.getCell(3, 3));

    // Corner cells
    bg.setCell(0, 0, true);
    bg.setCell(0, 7, true);
    bg.setCell(5, 0, true);
    bg.setCell(5, 7, true);
    EXPECT_TRUE(bg.getCell(0, 0));
    EXPECT_TRUE(bg.getCell(0, 7));
    EXPECT_TRUE(bg.getCell(5, 0));
    EXPECT_TRUE(bg.getCell(5, 7));

    // Setting alive on already-alive cell is idempotent
    bg.setCell(2, 3, true);
    EXPECT_TRUE(bg.getCell(2, 3));
    EXPECT_EQ(bg.aliveCells(), 5u);
}

TEST(BitGrid, SetCellClear) {
    BitGrid bg(4, 4);
    bg.setCell(1, 1, true);
    EXPECT_TRUE(bg.getCell(1, 1));
    bg.setCell(1, 1, false);
    EXPECT_FALSE(bg.getCell(1, 1));

    // Setting false on already-dead cell is idempotent
    bg.setCell(0, 0, false);
    EXPECT_FALSE(bg.getCell(0, 0));

    // Corner clear
    bg.setCell(3, 3, true);
    EXPECT_TRUE(bg.getCell(3, 3));
    bg.setCell(3, 3, false);
    EXPECT_FALSE(bg.getCell(3, 3));
}

TEST(BitGrid, WordBoundary) {
    // Cols = 130 → 3 words per row. Test cells at word boundaries.
    BitGrid bg(3, 130);
    bg.setCell(1, 63, true);  // last bit of word 0
    bg.setCell(1, 64, true);  // first bit of word 1
    bg.setCell(1, 127, true); // last bit of word 1
    bg.setCell(1, 128, true); // first bit of word 2

    EXPECT_TRUE(bg.getCell(1, 63));
    EXPECT_TRUE(bg.getCell(1, 64));
    EXPECT_TRUE(bg.getCell(1, 127));
    EXPECT_TRUE(bg.getCell(1, 128));
    EXPECT_FALSE(bg.getCell(1, 62));
    EXPECT_FALSE(bg.getCell(1, 65));
    EXPECT_FALSE(bg.getCell(1, 126));
    EXPECT_FALSE(bg.getCell(1, 129));
}

TEST(BitGrid, NonMultipleOf64Cols) {
    // 70 cols → 2 words per row, last word has 6 valid bits
    BitGrid bg(2, 70);
    bg.setCell(0, 69, true); // last valid column
    bg.setCell(1, 0, true);

    EXPECT_TRUE(bg.getCell(0, 69));
    EXPECT_TRUE(bg.getCell(1, 0));
    EXPECT_EQ(bg.aliveCells(), 2u);
}

TEST(BitGrid, AliveCellsCounting) {
    BitGrid bg(4, 4);
    EXPECT_EQ(bg.aliveCells(), 0u);

    bg.setCell(0, 0, true);
    EXPECT_EQ(bg.aliveCells(), 1u);

    bg.setCell(3, 3, true);
    bg.setCell(1, 2, true);
    EXPECT_EQ(bg.aliveCells(), 3u);

    bg.setCell(0, 0, false);
    EXPECT_EQ(bg.aliveCells(), 2u);
}

// -- Grid: file I/O ----------------------------------------------------------

TEST(Grid, LoadStillLifes) {
    Grid g(dataFile("still_lifes.txt"));
    ASSERT_EQ(g.getNumRows(), 10u);
    ASSERT_EQ(g.getNumCols(), 10u);

    EXPECT_TRUE(g.getCell(1, 1));
    EXPECT_TRUE(g.getCell(1, 2));
    EXPECT_FALSE(g.getCell(1, 0));
    EXPECT_FALSE(g.getCell(1, 3));

    EXPECT_TRUE(g.getCell(2, 0));
    EXPECT_TRUE(g.getCell(2, 3));
    EXPECT_TRUE(g.getCell(2, 6));
    EXPECT_TRUE(g.getCell(2, 7));
    EXPECT_FALSE(g.getCell(2, 1));
}

TEST(Grid, RandomInitAliveCount) {
    std::mt19937 rng(42);
    Grid g(100, 100, 500, rng);
    EXPECT_EQ(g.aliveCells(), 500u);
}

TEST(BitGrid, RandomInitAliveCount) {
    std::mt19937 rng(42);
    BitGrid bg(100, 100, 500, rng);
    EXPECT_EQ(bg.aliveCells(), 500u);
}

TEST(BitGrid, ConsistencyWithGrid) {
    const size_t rows = 64, cols = 128;
    const unsigned alive = 400;

    std::mt19937 rng1(99);
    Grid g(rows, cols, alive, rng1);

    std::mt19937 rng2(99);
    BitGrid bg(rows, cols, alive, rng2);

    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            ASSERT_EQ(g.getCell(r, c), bg.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

TEST(Grid, WriteReloadRoundTrip) {
    std::mt19937 rng(123);
    Grid original(30, 40, 100, rng);

    std::string tmpPath =
        (std::filesystem::temp_directory_path() / "gol_test_roundtrip.txt")
            .string();
    original.writeToFile(tmpPath);

    Grid reloaded(tmpPath);
    ASSERT_EQ(reloaded.getNumRows(), original.getNumRows());
    ASSERT_EQ(reloaded.getNumCols(), original.getNumCols());

    for (size_t r = 0; r < original.getNumRows(); r++)
        for (size_t c = 0; c < original.getNumCols(); c++)
            ASSERT_EQ(original.getCell(r, c), reloaded.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";

    std::remove(tmpPath.c_str());
}

// -- Grid: setRow / getRowData -----------------------------------------------

TEST(Grid, SetRowGetRowData) {
    Grid g(3, 4);
    uint8_t row1[] = {1, 0, 1, 0};
    g.setRow(1, row1);

    const uint8_t *data = g.getRowData(1);
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[1], 0);
    EXPECT_EQ(data[2], 1);
    EXPECT_EQ(data[3], 0);
    EXPECT_EQ(g.aliveCells(), 2u);
}

TEST(Grid, GetStride) {
    Grid g(5, 10);
    EXPECT_EQ(g.getStride(), 10u);
}

TEST(BitGrid, GetStride) {
    BitGrid bg(5, 70);
    // 70 cols → ceil(70/64) = 2 words per row
    EXPECT_EQ(bg.getStride(), 2u);

    BitGrid bg2(5, 128);
    EXPECT_EQ(bg2.getStride(), 2u);

    BitGrid bg3(5, 129);
    EXPECT_EQ(bg3.getStride(), 3u);
}
