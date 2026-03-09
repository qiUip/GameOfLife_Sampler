#include <gtest/gtest.h>

#include "gol_engine.h"
#include "grid.h"

#include <cstdint>
#include <string>
#include <vector>

static std::string dataFile(const char *name) {
    return std::string(TEST_DATA_DIR) + "/" + name;
}

// -- Helpers -----------------------------------------------------------------

// Read back byte engine state into a Grid
static Grid readBack(GameOfLife &engine, size_t rows, size_t cols) {
    Grid g(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        auto *row = static_cast<uint8_t *>(engine.getRowDataRaw(r));
        for (size_t c = 0; c < cols; c++)
            g.setCell(r, c, row[c]);
    }
    return g;
}

// Read back bitpack engine state into a BitGrid
static BitGrid readBackBit(GameOfLife &engine, size_t rows, size_t cols) {
    BitGrid bg(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        auto *words = static_cast<uint64_t *>(engine.getRowDataRaw(r));
        for (size_t c = 0; c < cols; c++)
            bg.setCell(r, c, (words[c / 64] >> (c % 64)) & 1);
    }
    return bg;
}

static std::vector<std::pair<size_t, size_t>> alivePositions(const Grid &g) {
    std::vector<std::pair<size_t, size_t>> cells;
    for (size_t r = 0; r < g.getNumRows(); r++)
        for (size_t c = 0; c < g.getNumCols(); c++)
            if (g.getCell(r, c))
                cells.emplace_back(r, c);
    return cells;
}

// =============================================================================
// SIMD unit tests for static helper functions
// =============================================================================

// -- golRule exhaustive -------------------------------------------------------

TEST(GolRule, DeadCellAllCounts) {
    for (uint8_t nb = 0; nb <= 8; ++nb) {
        uint8_t result = SIMDGameOfLife::golRule(0, nb);
        if (nb == 3)
            EXPECT_EQ(result, 1) << "dead + nb=" << (int)nb;
        else
            EXPECT_EQ(result, 0) << "dead + nb=" << (int)nb;
    }
}

TEST(GolRule, AliveCellAllCounts) {
    for (uint8_t nb = 0; nb <= 8; ++nb) {
        uint8_t result = SIMDGameOfLife::golRule(1, nb);
        if (nb == 2 || nb == 3)
            EXPECT_EQ(result, 1) << "alive + nb=" << (int)nb;
        else
            EXPECT_EQ(result, 0) << "alive + nb=" << (int)nb;
    }
}

// -- aliveNeighbours ----------------------------------------------------------

// Interior: all 8 neighbours present. 3×5 rows, query col 2.
TEST(AliveNeighbours, Interior) {
    //        col: 0  1  2  3  4
    uint8_t p[] = {1, 0, 1, 1, 0};
    uint8_t c[] = {0, 1, 0, 1, 0};
    uint8_t n[] = {0, 0, 1, 0, 0};
    // neighbours of col 2: p[1]=0 p[2]=1 p[3]=1 c[1]=1 c[3]=1 n[1]=0 n[2]=1
    // n[3]=0
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<true, true, true, true>(p, c, n, 2)),
        5);
}

TEST(AliveNeighbours, TopLeftCorner) {
    uint8_t p[] = {0}; // unused
    uint8_t c[] = {1, 1, 0};
    uint8_t n[] = {1, 1, 0};
    // col 0, no prev, no left → neighbours: c[1]=1 n[0]=1 n[1]=1
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<false, true, false, true>(p, c, n, 0)),
        3);
}

TEST(AliveNeighbours, TopRightCorner) {
    uint8_t p[] = {0};
    uint8_t c[] = {0, 1, 1};
    uint8_t n[] = {0, 1, 0};
    // col 2, no prev, no right → neighbours: c[1]=1 n[1]=1 n[2]=0
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<false, true, true, false>(p, c, n, 2)),
        2);
}

TEST(AliveNeighbours, BottomLeftCorner) {
    uint8_t p[] = {1, 1, 0};
    uint8_t c[] = {0, 1, 0};
    uint8_t n[] = {0};
    // col 0, no next, no left → neighbours: p[0]=1 p[1]=1 c[1]=1
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<true, false, false, true>(p, c, n, 0)),
        3);
}

TEST(AliveNeighbours, BottomRightCorner) {
    uint8_t p[] = {0, 1, 1};
    uint8_t c[] = {0, 1, 0};
    uint8_t n[] = {0};
    // col 2, no next, no right → neighbours: p[1]=1 p[2]=1 c[1]=1
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<true, false, true, false>(p, c, n, 2)),
        3);
}

TEST(AliveNeighbours, TopEdge) {
    uint8_t p[] = {0};
    uint8_t c[] = {0, 1, 0, 1, 0};
    uint8_t n[] = {1, 1, 1, 0, 0};
    // col 2, no prev → neighbours: c[1]=1 c[3]=1 n[1]=1 n[2]=1 n[3]=0
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<false, true, true, true>(p, c, n, 2)),
        4);
}

TEST(AliveNeighbours, BottomEdge) {
    uint8_t p[] = {0, 1, 1, 1, 0};
    uint8_t c[] = {0, 0, 0, 0, 0};
    uint8_t n[] = {0};
    // col 2, no next → neighbours: p[1]=1 p[2]=1 p[3]=1 c[1]=0 c[3]=0
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<true, false, true, true>(p, c, n, 2)),
        3);
}

TEST(AliveNeighbours, LeftEdge) {
    uint8_t p[] = {1, 1, 0};
    uint8_t c[] = {0, 1, 0};
    uint8_t n[] = {1, 0, 0};
    // col 0, no left → neighbours: p[0]=1 p[1]=1 c[1]=1 n[0]=1 n[1]=0
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<true, true, false, true>(p, c, n, 0)),
        4);
}

TEST(AliveNeighbours, RightEdge) {
    uint8_t p[] = {0, 1, 1};
    uint8_t c[] = {0, 0, 0};
    uint8_t n[] = {0, 1, 1};
    // col 2, no right → neighbours: p[1]=1 p[2]=1 c[1]=0 n[1]=1 n[2]=1
    EXPECT_EQ(
        (SIMDGameOfLife::aliveNeighbours<true, true, true, false>(p, c, n, 2)),
        4);
}

// -- processInteriorCells -----------------------------------------------------

TEST(ProcessInteriorCells, KnownOutput) {
    // 3 rows × 8 cols. Interior columns are [1..6].
    const size_t cols = 8;
    //         col: 0  1  2  3  4  5  6  7
    uint8_t p[]  = {0, 1, 1, 0, 0, 0, 0, 0};
    uint8_t c[]  = {0, 0, 1, 0, 0, 0, 0, 0};
    uint8_t n[]  = {0, 1, 1, 0, 0, 0, 0, 0};
    uint8_t o[8] = {};

    SIMDGameOfLife::processInteriorCells(p, c, n, o, cols);

    // col 1: nb = p[0]+p[1]+p[2] + c[0]+c[2] + n[0]+n[1]+n[2]
    //       = 0+1+1 + 0+1 + 0+1+1 = 5, c[1]=0 → dead
    EXPECT_EQ(o[1], 0);
    // col 2: nb = p[1]+p[2]+p[3] + c[1]+c[3] + n[1]+n[2]+n[3]
    //       = 1+1+0 + 0+0 + 1+1+0 = 4, c[2]=1 → dies
    EXPECT_EQ(o[2], 0);
    // col 3: nb = p[2]+p[3]+p[4] + c[2]+c[4] + n[2]+n[3]+n[4]
    //       = 1+0+0 + 1+0 + 1+0+0 = 3, c[3]=0 → born
    EXPECT_EQ(o[3], 1);
    // cols 4-6: all zero neighbours → stay dead
    for (size_t i = 4; i <= 6; ++i)
        EXPECT_EQ(o[i], 0) << "col " << i;
}

// -- processBorderRow ---------------------------------------------------------

TEST(ProcessBorderRow, TopRow) {
    const size_t cols = 5;
    uint8_t p[]       = {0}; // unused
    //         col: 0  1  2  3  4
    uint8_t c[]  = {0, 1, 1, 1, 0};
    uint8_t n[]  = {0, 0, 1, 0, 0};
    uint8_t o[5] = {};

    SIMDGameOfLife::processBorderRow<false, true>(p, c, n, o, cols);

    // col 0: nb = c[1] + n[0] + n[1] = 1+0+0 = 1, dead → 0
    EXPECT_EQ(o[0], 0);
    // col 1: nb = c[0] + c[2] + n[0] + n[1] + n[2] = 0+1+0+0+1 = 2, alive →
    // survive
    EXPECT_EQ(o[1], 1);
    // col 2: nb = c[1] + c[3] + n[1] + n[2] + n[3] = 1+1+0+1+0 = 3, alive →
    // survive
    EXPECT_EQ(o[2], 1);
    // col 3: nb = c[2] + c[4] + n[2] + n[3] + n[4] = 1+0+1+0+0 = 2, alive →
    // survive
    EXPECT_EQ(o[3], 1);
    // col 4: nb = c[3] + n[3] + n[4] = 1+0+0 = 1, dead → 0
    EXPECT_EQ(o[4], 0);
}

TEST(ProcessBorderRow, BottomRow) {
    const size_t cols = 5;
    //                  col: 0  1  2  3  4
    uint8_t p[]  = {0, 0, 1, 0, 0};
    uint8_t c[]  = {0, 1, 1, 1, 0};
    uint8_t n[]  = {0}; // unused
    uint8_t o[5] = {};

    SIMDGameOfLife::processBorderRow<true, false>(p, c, n, o, cols);

    EXPECT_EQ(o[0], 0);
    // col 1: nb = p[0]+p[1]+p[2] + c[0]+c[2] = 0+0+1+0+1 = 2, alive → survive
    EXPECT_EQ(o[1], 1);
    // col 2: nb = p[1]+p[2]+p[3] + c[1]+c[3] = 0+1+0+1+1 = 3, alive → survive
    EXPECT_EQ(o[2], 1);
    // col 3: nb = p[2]+p[3]+p[4] + c[2]+c[4] = 1+0+0+1+0 = 2, alive → survive
    EXPECT_EQ(o[3], 1);
    EXPECT_EQ(o[4], 0);
}

// -- processInteriorRow -------------------------------------------------------

TEST(ProcessInteriorRow, FullRow) {
    const size_t cols = 6;
    //                  col: 0  1  2  3  4  5
    uint8_t p[]  = {0, 1, 1, 0, 0, 0};
    uint8_t c[]  = {1, 0, 1, 0, 0, 0};
    uint8_t n[]  = {0, 1, 1, 0, 0, 0};
    uint8_t o[6] = {};

    SIMDGameOfLife::processInteriorRow(p, c, n, o, cols);

    // col 0 (left border, no left neighbour):
    //   nb = p[0]+p[1] + c[1] + n[0]+n[1] = 0+1+0+0+1 = 2, c[0]=1 → survive
    EXPECT_EQ(o[0], 1);
    // col 1 (interior): nb = p[0]+p[1]+p[2]+c[0]+c[2]+n[0]+n[1]+n[2]
    //   = 0+1+1+1+1+0+1+1 = 6, c[1]=0 → dead
    EXPECT_EQ(o[1], 0);
    // col 2 (interior): nb = p[1]+p[2]+p[3]+c[1]+c[3]+n[1]+n[2]+n[3]
    //   = 1+1+0+0+0+1+1+0 = 4, c[2]=1 → dies
    EXPECT_EQ(o[2], 0);
    // col 3 (interior): nb = p[2]+p[3]+p[4]+c[2]+c[4]+n[2]+n[3]+n[4]
    //   = 1+0+0+1+0+1+0+0 = 3, c[3]=0 → born
    EXPECT_EQ(o[3], 1);
    // col 4 (interior): all zero around → 0
    EXPECT_EQ(o[4], 0);
    // col 5 (right border, no right neighbour):
    //   nb = p[4]+p[5] + c[4] + n[4]+n[5] = 0+0+0+0+0 = 0, dead → 0
    EXPECT_EQ(o[5], 0);
}

// =============================================================================
// BitPack unit tests for static helper functions
// =============================================================================

// -- rowSum3 ------------------------------------------------------------------

TEST(RowSum3, AllZeros) {
    uint64_t s1, s0;
    BitPackGameOfLife::rowSum3(0, 0, 0, s1, s0);
    EXPECT_EQ(s1, 0u);
    EXPECT_EQ(s0, 0u);
}

TEST(RowSum3, AllOnes) {
    uint64_t s1, s0;
    BitPackGameOfLife::rowSum3(~uint64_t(0), ~uint64_t(0), ~uint64_t(0), s1,
                               s0);
    // sum=3 → binary 11 → s1=all-ones, s0=all-ones
    EXPECT_EQ(s1, ~uint64_t(0));
    EXPECT_EQ(s0, ~uint64_t(0));
}

TEST(RowSum3, TwoOfThree) {
    uint64_t s1, s0;
    uint64_t all = ~uint64_t(0);
    BitPackGameOfLife::rowSum3(all, all, 0, s1, s0);
    // sum=2 → binary 10 → s1=all-ones, s0=0
    EXPECT_EQ(s1, all);
    EXPECT_EQ(s0, 0u);
}

TEST(RowSum3, OneOfThree) {
    uint64_t s1, s0;
    BitPackGameOfLife::rowSum3(~uint64_t(0), 0, 0, s1, s0);
    // sum=1 → binary 01 → s1=0, s0=all-ones
    EXPECT_EQ(s1, 0u);
    EXPECT_EQ(s0, ~uint64_t(0));
}

TEST(RowSum3, MixedBits) {
    // bit 0: L=1,C=0,R=1 → sum=2 → s1[0]=1, s0[0]=0
    // bit 1: L=0,C=1,R=0 → sum=1 → s1[1]=0, s0[1]=1
    // bit 2: L=1,C=1,R=1 → sum=3 → s1[2]=1, s0[2]=1
    // bit 3: L=0,C=0,R=0 → sum=0 → s1[3]=0, s0[3]=0
    uint64_t L = 0b0101;
    uint64_t C = 0b0110;
    uint64_t R = 0b0101;
    uint64_t s1, s0;
    BitPackGameOfLife::rowSum3(L, C, R, s1, s0);
    EXPECT_EQ(s1 & 0xF, 0b0101u); // bits 0,2
    EXPECT_EQ(s0 & 0xF, 0b0110u); // bits 1,2
}

// -- sum9 ---------------------------------------------------------------------

TEST(Sum9, AllZeros) {
    uint64_t o3, o2, o1, o0;
    BitPackGameOfLife::sum9(0, 0, 0, 0, 0, 0, o3, o2, o1, o0);
    EXPECT_EQ(o3, 0u);
    EXPECT_EQ(o2, 0u);
    EXPECT_EQ(o1, 0u);
    EXPECT_EQ(o0, 0u);
}

TEST(Sum9, SumThree) {
    // Each row-sum = 1 (binary 01), three rows → total 3 (binary 0011)
    uint64_t all = ~uint64_t(0);
    uint64_t o3, o2, o1, o0;
    BitPackGameOfLife::sum9(0, all, 0, all, 0, all, o3, o2, o1, o0);
    EXPECT_EQ(o3, 0u);
    EXPECT_EQ(o2, 0u);
    EXPECT_EQ(o1, all);
    EXPECT_EQ(o0, all);
}

TEST(Sum9, SumFour) {
    // Row-sums: 2 (10) + 1 (01) + 1 (01) = 4 (0100)
    uint64_t all = ~uint64_t(0);
    uint64_t o3, o2, o1, o0;
    BitPackGameOfLife::sum9(all, 0, 0, all, 0, all, o3, o2, o1, o0);
    EXPECT_EQ(o3, 0u);
    EXPECT_EQ(o2, all);
    EXPECT_EQ(o1, 0u);
    EXPECT_EQ(o0, 0u);
}

TEST(Sum9, SumNine) {
    // Each row-sum = 3 (11), three rows → total 9 (1001)
    uint64_t all = ~uint64_t(0);
    uint64_t o3, o2, o1, o0;
    BitPackGameOfLife::sum9(all, all, all, all, all, all, o3, o2, o1, o0);
    EXPECT_EQ(o3, all);
    EXPECT_EQ(o2, 0u);
    EXPECT_EQ(o1, 0u);
    EXPECT_EQ(o0, all);
}

// =============================================================================
// SIMD takeStep tests
// =============================================================================

// Use a wide grid to ensure processInteriorCells is exercised (cols >> SIMD
// width)
TEST(SIMD, InteriorCellsWideGrid) {
    // 3 rows × 256 cols. Place a horizontal line of 3 at row 1, cols 100-102.
    // After 1 step: cell (0,101) born, (2,101) born, (1,100) and (1,102) die,
    // (1,101) survives → vertical blinker at col 101.
    const size_t rows = 3, cols = 256;
    Grid g(rows, cols);
    g.setCell(1, 100, true);
    g.setCell(1, 101, true);
    g.setCell(1, 102, true);
    SIMDGameOfLife engine(g);
    engine.takeStep();
    Grid after = readBack(engine, rows, cols);

    EXPECT_FALSE(after.getCell(1, 100));
    EXPECT_TRUE(after.getCell(1, 101));
    EXPECT_FALSE(after.getCell(1, 102));
    EXPECT_TRUE(after.getCell(0, 101));
    EXPECT_TRUE(after.getCell(2, 101));
    EXPECT_EQ(after.aliveCells(), 3u);
}

// SIMD with scalar tail: cols such that interior isn't a multiple of SIMD width
TEST(SIMD, ScalarTailHandling) {
    // 5 rows × 19 cols — small enough that SIMD loop body runs 0 or 1 times
    // with a scalar tail. Place blinker at row 2, cols 8-10.
    const size_t rows = 5, cols = 19;
    Grid g(rows, cols);
    g.setCell(2, 8, true);
    g.setCell(2, 9, true);
    g.setCell(2, 10, true);
    SIMDGameOfLife engine(g);
    engine.takeStep();
    Grid after = readBack(engine, rows, cols);

    EXPECT_TRUE(after.getCell(1, 9));
    EXPECT_TRUE(after.getCell(2, 9));
    EXPECT_TRUE(after.getCell(3, 9));
    EXPECT_FALSE(after.getCell(2, 8));
    EXPECT_FALSE(after.getCell(2, 10));
}

// =============================================================================
// BitPack takeStep tests
// =============================================================================

TEST(BitPack, Birth) {
    BitGrid bg(5, 5);
    bg.setCell(1, 1, true);
    bg.setCell(1, 2, true);
    bg.setCell(1, 3, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2))
        << "dead cell with 3 neighbours should be born";
}

TEST(BitPack, Underpopulation) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2)) << "alone cell should die";
}

TEST(BitPack, Overcrowding) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    bg.setCell(1, 1, true);
    bg.setCell(1, 2, true);
    bg.setCell(1, 3, true);
    bg.setCell(2, 1, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2)) << "cell with 4 neighbours should die";
}

TEST(BitPack, Survival) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    bg.setCell(1, 1, true);
    bg.setCell(1, 3, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2)) << "cell with 2 neighbours should survive";
}

// BitPack word boundary: blinker straddling cols 63-65
TEST(BitPack, WordBoundaryBlinker) {
    BitGrid bg(5, 130);
    bg.setCell(2, 63, true);
    bg.setCell(2, 64, true);
    bg.setCell(2, 65, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 130);

    // Horizontal blinker → vertical after 1 step
    EXPECT_FALSE(after.getCell(2, 63));
    EXPECT_TRUE(after.getCell(2, 64));
    EXPECT_FALSE(after.getCell(2, 65));
    EXPECT_TRUE(after.getCell(1, 64));
    EXPECT_TRUE(after.getCell(3, 64));
}

// BitPack: non-multiple-of-64 cols, cells near last valid column
TEST(BitPack, NonMultiple64Cols) {
    // 70 cols. Blinker at row 2, cols 67-69 (near end of valid bits).
    BitGrid bg(5, 70);
    bg.setCell(2, 67, true);
    bg.setCell(2, 68, true);
    bg.setCell(2, 69, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 70);

    // After 1 step: vertical at col 68, but (2,69) is right edge so only
    // (2,68) has 2 neighbours → survive, (1,68) and (3,68) born from 3 each.
    // (2,67) has 1 neighbour → dies. (2,69) has 1 neighbour → dies.
    EXPECT_FALSE(after.getCell(2, 67));
    EXPECT_TRUE(after.getCell(2, 68));
    EXPECT_FALSE(after.getCell(2, 69));
    EXPECT_TRUE(after.getCell(1, 68));
    EXPECT_TRUE(after.getCell(3, 68));
}

// BitPack corner handling
TEST(BitPack, CornerBirth) {
    BitGrid bg(5, 5);
    bg.setCell(0, 1, true);
    bg.setCell(1, 0, true);
    bg.setCell(1, 1, true);
    BitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(0, 0))
        << "corner (0,0) with 3 neighbours should be born";
}

// =============================================================================
// GoL rule tests via takeStep on crafted grids (Simple + SIMD)
// =============================================================================

// Dead cell with exactly 3 neighbours → born
template <typename EngineT> void testBirth() {
    // 5×5 grid, 3 alive cells around (2,2):
    //   (1,1) (1,2) (1,3) alive → cell (2,2) has 3 neighbours → born
    Grid g(5, 5);
    g.setCell(1, 1, true);
    g.setCell(1, 2, true);
    g.setCell(1, 3, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2))
        << "dead cell with 3 neighbours should be born";
}

TEST(Rule, Birth_Simple) {
    testBirth<SimpleGameOfLife>();
}
TEST(Rule, Birth_SIMD) {
    testBirth<SIMDGameOfLife>();
}

// Alive cell with 2 neighbours → survives
template <typename EngineT> void testSurvival2() {
    // 5×5: alive (2,2) with neighbours (1,1) and (1,3) → 2 neighbours → survive
    Grid g(5, 5);
    g.setCell(2, 2, true);
    g.setCell(1, 1, true);
    g.setCell(1, 3, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2))
        << "alive cell with 2 neighbours should survive";
}

TEST(Rule, Survival2_Simple) {
    testSurvival2<SimpleGameOfLife>();
}
TEST(Rule, Survival2_SIMD) {
    testSurvival2<SIMDGameOfLife>();
}

// Alive cell with 3 neighbours → survives
template <typename EngineT> void testSurvival3() {
    // alive (2,2) with neighbours (1,1) (1,2) (1,3)
    Grid g(5, 5);
    g.setCell(2, 2, true);
    g.setCell(1, 1, true);
    g.setCell(1, 2, true);
    g.setCell(1, 3, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2))
        << "alive cell with 3 neighbours should survive";
}

TEST(Rule, Survival3_Simple) {
    testSurvival3<SimpleGameOfLife>();
}
TEST(Rule, Survival3_SIMD) {
    testSurvival3<SIMDGameOfLife>();
}

// Alive cell with 0 neighbours → dies (underpopulation)
template <typename EngineT> void testUnderpopulation0() {
    Grid g(5, 5);
    g.setCell(2, 2, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2))
        << "alive cell with 0 neighbours should die";
}

TEST(Rule, Underpopulation0_Simple) {
    testUnderpopulation0<SimpleGameOfLife>();
}
TEST(Rule, Underpopulation0_SIMD) {
    testUnderpopulation0<SIMDGameOfLife>();
}

// Alive cell with 1 neighbour → dies (underpopulation)
template <typename EngineT> void testUnderpopulation1() {
    Grid g(5, 5);
    g.setCell(2, 2, true);
    g.setCell(1, 1, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2))
        << "alive cell with 1 neighbour should die";
}

TEST(Rule, Underpopulation1_Simple) {
    testUnderpopulation1<SimpleGameOfLife>();
}
TEST(Rule, Underpopulation1_SIMD) {
    testUnderpopulation1<SIMDGameOfLife>();
}

// Alive cell with 4 neighbours → dies (overcrowding)
template <typename EngineT> void testOvercrowding4() {
    Grid g(5, 5);
    g.setCell(2, 2, true);
    g.setCell(1, 1, true);
    g.setCell(1, 2, true);
    g.setCell(1, 3, true);
    g.setCell(2, 1, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2))
        << "alive cell with 4 neighbours should die";
}

TEST(Rule, Overcrowding4_Simple) {
    testOvercrowding4<SimpleGameOfLife>();
}
TEST(Rule, Overcrowding4_SIMD) {
    testOvercrowding4<SIMDGameOfLife>();
}

// Dead cell with 2 neighbours → stays dead
template <typename EngineT> void testDeadStaysDead() {
    Grid g(5, 5);
    g.setCell(1, 1, true);
    g.setCell(1, 3, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2))
        << "dead cell with 2 neighbours should stay dead";
}

TEST(Rule, DeadStaysDead_Simple) {
    testDeadStaysDead<SimpleGameOfLife>();
}
TEST(Rule, DeadStaysDead_SIMD) {
    testDeadStaysDead<SIMDGameOfLife>();
}

// Dead cell with 4 neighbours → stays dead
template <typename EngineT> void testDeadWith4() {
    Grid g(5, 5);
    g.setCell(1, 1, true);
    g.setCell(1, 2, true);
    g.setCell(1, 3, true);
    g.setCell(2, 1, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2))
        << "dead cell with 4 neighbours should stay dead";
}

TEST(Rule, DeadWith4_Simple) {
    testDeadWith4<SimpleGameOfLife>();
}
TEST(Rule, DeadWith4_SIMD) {
    testDeadWith4<SIMDGameOfLife>();
}

// -- Boundary cell tests -----------------------------------------------------

// Top-left corner: alive with 1 neighbour → dies
template <typename EngineT> void testCornerTopLeft() {
    Grid g(5, 5);
    g.setCell(0, 0, true);
    g.setCell(0, 1, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(0, 0))
        << "corner (0,0) with 1 neighbour should die";
}

TEST(Boundary, CornerTopLeft_Simple) {
    testCornerTopLeft<SimpleGameOfLife>();
}
TEST(Boundary, CornerTopLeft_SIMD) {
    testCornerTopLeft<SIMDGameOfLife>();
}

// Top-right corner: birth from 3 neighbours
template <typename EngineT> void testCornerTopRight() {
    Grid g(5, 8);
    // Make (0,7) dead, give it 3 neighbours: (0,6) (1,6) (1,7)
    g.setCell(0, 6, true);
    g.setCell(1, 6, true);
    g.setCell(1, 7, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 8);
    EXPECT_TRUE(after.getCell(0, 7))
        << "corner (0,7) with 3 neighbours should be born";
}

TEST(Boundary, CornerTopRight_Simple) {
    testCornerTopRight<SimpleGameOfLife>();
}
TEST(Boundary, CornerTopRight_SIMD) {
    testCornerTopRight<SIMDGameOfLife>();
}

// Bottom-left corner: birth from 3 neighbours
template <typename EngineT> void testCornerBottomLeft() {
    Grid g(5, 5);
    g.setCell(3, 0, true);
    g.setCell(3, 1, true);
    g.setCell(4, 1, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_TRUE(after.getCell(4, 0))
        << "corner (4,0) with 3 neighbours should be born";
}

TEST(Boundary, CornerBottomLeft_Simple) {
    testCornerBottomLeft<SimpleGameOfLife>();
}
TEST(Boundary, CornerBottomLeft_SIMD) {
    testCornerBottomLeft<SIMDGameOfLife>();
}

// Bottom-right corner: survival with 2 neighbours
template <typename EngineT> void testCornerBottomRight() {
    Grid g(5, 5);
    g.setCell(4, 4, true);
    g.setCell(3, 3, true);
    g.setCell(3, 4, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_TRUE(after.getCell(4, 4))
        << "corner (4,4) with 2 neighbours should survive";
}

TEST(Boundary, CornerBottomRight_Simple) {
    testCornerBottomRight<SimpleGameOfLife>();
}
TEST(Boundary, CornerBottomRight_SIMD) {
    testCornerBottomRight<SIMDGameOfLife>();
}

// Top edge (not corner): cell (0,3) with 3 neighbours → born
template <typename EngineT> void testTopEdge() {
    Grid g(5, 8);
    g.setCell(0, 2, true);
    g.setCell(0, 4, true);
    g.setCell(1, 3, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 8);
    EXPECT_TRUE(after.getCell(0, 3))
        << "top edge cell with 3 neighbours should be born";
}

TEST(Boundary, TopEdge_Simple) {
    testTopEdge<SimpleGameOfLife>();
}
TEST(Boundary, TopEdge_SIMD) {
    testTopEdge<SIMDGameOfLife>();
}

// Left edge (not corner): cell (2,0) with 3 neighbours → born
template <typename EngineT> void testLeftEdge() {
    Grid g(5, 5);
    g.setCell(1, 0, true);
    g.setCell(1, 1, true);
    g.setCell(3, 0, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 0))
        << "left edge cell with 3 neighbours should be born";
}

TEST(Boundary, LeftEdge_Simple) {
    testLeftEdge<SimpleGameOfLife>();
}
TEST(Boundary, LeftEdge_SIMD) {
    testLeftEdge<SIMDGameOfLife>();
}

// Right edge: cell (2, last) with 3 neighbours → born
template <typename EngineT> void testRightEdge() {
    Grid g(5, 8);
    g.setCell(1, 7, true);
    g.setCell(1, 6, true);
    g.setCell(3, 7, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 8);
    EXPECT_TRUE(after.getCell(2, 7))
        << "right edge cell with 3 neighbours should be born";
}

TEST(Boundary, RightEdge_Simple) {
    testRightEdge<SimpleGameOfLife>();
}
TEST(Boundary, RightEdge_SIMD) {
    testRightEdge<SIMDGameOfLife>();
}

// Bottom edge: cell (last, 3) with 3 neighbours → born
template <typename EngineT> void testBottomEdge() {
    Grid g(5, 8);
    g.setCell(4, 2, true);
    g.setCell(4, 4, true);
    g.setCell(3, 3, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 8);
    EXPECT_TRUE(after.getCell(4, 3))
        << "bottom edge cell with 3 neighbours should be born";
}

TEST(Boundary, BottomEdge_Simple) {
    testBottomEdge<SimpleGameOfLife>();
}
TEST(Boundary, BottomEdge_SIMD) {
    testBottomEdge<SIMDGameOfLife>();
}

// =============================================================================
// Multi-step / pattern tests (all engines)
// =============================================================================

// -- Still lifes -------------------------------------------------------------

template <typename EngineT> void testStillLifesStable() {
    Grid before(dataFile("still_lifes.txt"));
    auto expected = alivePositions(before);

    Grid g(dataFile("still_lifes.txt"));
    EngineT engine(g);
    for (int i = 0; i < 10; i++)
        engine.takeStep();

    Grid after = readBack(engine, before.getNumRows(), before.getNumCols());
    ASSERT_EQ(alivePositions(after), expected);
}

TEST(Engine, StillLifes_Simple) {
    testStillLifesStable<SimpleGameOfLife>();
}
TEST(Engine, StillLifes_SIMD) {
    testStillLifesStable<SIMDGameOfLife>();
}

TEST(Engine, StillLifes_BitPack) {
    BitGrid before(dataFile("still_lifes.txt"));
    size_t rows = before.getNumRows(), cols = before.getNumCols();

    BitGrid bg(dataFile("still_lifes.txt"));
    BitPackGameOfLife engine(bg);
    for (int i = 0; i < 10; i++)
        engine.takeStep();

    BitGrid after = readBackBit(engine, rows, cols);
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            ASSERT_EQ(after.getCell(r, c), before.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

// -- Blinker period-2 --------------------------------------------------------

template <typename EngineT> void testBlinkerPeriod2() {
    Grid g(5, 5);
    g.setCell(2, 1, true);
    g.setCell(2, 2, true);
    g.setCell(2, 3, true);
    auto expected = alivePositions(g);
    EngineT engine(g);
    engine.takeStep();
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    ASSERT_EQ(alivePositions(after), expected);
}

TEST(Engine, Blinker_Simple) {
    testBlinkerPeriod2<SimpleGameOfLife>();
}
TEST(Engine, Blinker_SIMD) {
    testBlinkerPeriod2<SIMDGameOfLife>();
}

// -- Glider displacement -----------------------------------------------------

TEST(Engine, GliderDisplacement_Simple) {
    Grid ref(dataFile("glider.txt"));
    size_t rows = ref.getNumRows(), cols = ref.getNumCols();
    auto before = alivePositions(ref);
    ASSERT_EQ(before.size(), 5u);

    Grid g(dataFile("glider.txt"));
    SimpleGameOfLife engine(g);
    for (int i = 0; i < 4; i++)
        engine.takeStep();

    Grid after      = readBack(engine, rows, cols);
    auto afterCells = alivePositions(after);
    ASSERT_EQ(afterCells.size(), 5u);

    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(afterCells[i].first, before[i].first + 1);
        EXPECT_EQ(afterCells[i].second, before[i].second + 1);
    }
}

// -- All-dead stays dead -----------------------------------------------------

TEST(Engine, AllDeadStaysDead_Simple) {
    Grid g(10, 10);
    SimpleGameOfLife engine(g);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    Grid after = readBack(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}

TEST(Engine, AllDeadStaysDead_SIMD) {
    Grid g(10, 10);
    SIMDGameOfLife engine(g);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    Grid after = readBack(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}

TEST(Engine, AllDeadStaysDead_BitPack) {
    BitGrid bg(10, 10);
    BitPackGameOfLife engine(bg);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    BitGrid after = readBackBit(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}
