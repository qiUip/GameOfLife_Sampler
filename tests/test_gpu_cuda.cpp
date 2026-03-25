#if GOL_CUDA

#include <gtest/gtest.h>

#include "gol_gpu.h"
#include "gol_gpu_test_wrappers.h"
#include "grid.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static std::string dataFile(const char *name) {
    return std::string(TEST_DATA_DIR) + "/" + name;
}

// -- Helpers -----------------------------------------------------------------

static Grid readBack(GameOfLife &engine, size_t rows, size_t cols) {
    Grid g(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        auto *row = static_cast<uint8_t *>(engine.getRowDataRaw(r));
        for (size_t c = 0; c < cols; c++)
            g.setCell(r, c, row[c]);
    }
    return g;
}

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
// Engine-class tests (byte engines: Simple + Tile)
// =============================================================================

// -- GoL rule tests -----------------------------------------------------------

template <typename EngineT> void testBirth() {
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

TEST(CUDARule, Birth_Simple) {
    testBirth<CUDASimpleGameOfLife>();
}
TEST(CUDARule, Birth_Tile) {
    testBirth<CUDATileGameOfLife>();
}

template <typename EngineT> void testSurvival2() {
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

TEST(CUDARule, Survival2_Simple) {
    testSurvival2<CUDASimpleGameOfLife>();
}
TEST(CUDARule, Survival2_Tile) {
    testSurvival2<CUDATileGameOfLife>();
}

template <typename EngineT> void testSurvival3() {
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

TEST(CUDARule, Survival3_Simple) {
    testSurvival3<CUDASimpleGameOfLife>();
}
TEST(CUDARule, Survival3_Tile) {
    testSurvival3<CUDATileGameOfLife>();
}

template <typename EngineT> void testUnderpopulation0() {
    Grid g(5, 5);
    g.setCell(2, 2, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2))
        << "alive cell with 0 neighbours should die";
}

TEST(CUDARule, Underpop0_Simple) {
    testUnderpopulation0<CUDASimpleGameOfLife>();
}
TEST(CUDARule, Underpop0_Tile) {
    testUnderpopulation0<CUDATileGameOfLife>();
}

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

TEST(CUDARule, Underpop1_Simple) {
    testUnderpopulation1<CUDASimpleGameOfLife>();
}
TEST(CUDARule, Underpop1_Tile) {
    testUnderpopulation1<CUDATileGameOfLife>();
}

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

TEST(CUDARule, Overcrowding_Simple) {
    testOvercrowding4<CUDASimpleGameOfLife>();
}
TEST(CUDARule, Overcrowding_Tile) {
    testOvercrowding4<CUDATileGameOfLife>();
}

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

TEST(CUDARule, DeadStaysDead_Simple) {
    testDeadStaysDead<CUDASimpleGameOfLife>();
}
TEST(CUDARule, DeadStaysDead_Tile) {
    testDeadStaysDead<CUDATileGameOfLife>();
}

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

TEST(CUDARule, DeadWith4_Simple) {
    testDeadWith4<CUDASimpleGameOfLife>();
}
TEST(CUDARule, DeadWith4_Tile) {
    testDeadWith4<CUDATileGameOfLife>();
}

// -- Boundary tests -----------------------------------------------------------

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

TEST(CUDABoundary, CornerTopLeft_Simple) {
    testCornerTopLeft<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, CornerTopLeft_Tile) {
    testCornerTopLeft<CUDATileGameOfLife>();
}

template <typename EngineT> void testCornerTopRight() {
    Grid g(5, 8);
    g.setCell(0, 6, true);
    g.setCell(1, 6, true);
    g.setCell(1, 7, true);
    EngineT engine(g);
    engine.takeStep();
    Grid after = readBack(engine, 5, 8);
    EXPECT_TRUE(after.getCell(0, 7))
        << "corner (0,7) with 3 neighbours should be born";
}

TEST(CUDABoundary, CornerTopRight_Simple) {
    testCornerTopRight<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, CornerTopRight_Tile) {
    testCornerTopRight<CUDATileGameOfLife>();
}

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

TEST(CUDABoundary, CornerBottomLeft_Simple) {
    testCornerBottomLeft<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, CornerBottomLeft_Tile) {
    testCornerBottomLeft<CUDATileGameOfLife>();
}

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

TEST(CUDABoundary, CornerBottomRight_Simple) {
    testCornerBottomRight<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, CornerBottomRight_Tile) {
    testCornerBottomRight<CUDATileGameOfLife>();
}

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

TEST(CUDABoundary, TopEdge_Simple) {
    testTopEdge<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, TopEdge_Tile) {
    testTopEdge<CUDATileGameOfLife>();
}

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

TEST(CUDABoundary, LeftEdge_Simple) {
    testLeftEdge<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, LeftEdge_Tile) {
    testLeftEdge<CUDATileGameOfLife>();
}

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

TEST(CUDABoundary, RightEdge_Simple) {
    testRightEdge<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, RightEdge_Tile) {
    testRightEdge<CUDATileGameOfLife>();
}

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

TEST(CUDABoundary, BottomEdge_Simple) {
    testBottomEdge<CUDASimpleGameOfLife>();
}
TEST(CUDABoundary, BottomEdge_Tile) {
    testBottomEdge<CUDATileGameOfLife>();
}

// -- Pattern tests ------------------------------------------------------------

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

TEST(CUDAPattern, StillLifes_Simple) {
    testStillLifesStable<CUDASimpleGameOfLife>();
}
TEST(CUDAPattern, StillLifes_Tile) {
    testStillLifesStable<CUDATileGameOfLife>();
}

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

TEST(CUDAPattern, Blinker_Simple) {
    testBlinkerPeriod2<CUDASimpleGameOfLife>();
}
TEST(CUDAPattern, Blinker_Tile) {
    testBlinkerPeriod2<CUDATileGameOfLife>();
}

template <typename EngineT> void testAllDeadStaysDead() {
    Grid g(10, 10);
    EngineT engine(g);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    Grid after = readBack(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}

TEST(CUDAPattern, AllDead_Simple) {
    testAllDeadStaysDead<CUDASimpleGameOfLife>();
}
TEST(CUDAPattern, AllDead_Tile) {
    testAllDeadStaysDead<CUDATileGameOfLife>();
}

template <typename EngineT> void testGliderDisplacement() {
    Grid ref(dataFile("glider.txt"));
    size_t rows = ref.getNumRows(), cols = ref.getNumCols();
    auto before = alivePositions(ref);
    ASSERT_EQ(before.size(), 5u);

    Grid g(dataFile("glider.txt"));
    EngineT engine(g);
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

TEST(CUDAPattern, Glider_Simple) {
    testGliderDisplacement<CUDASimpleGameOfLife>();
}
TEST(CUDAPattern, Glider_Tile) {
    testGliderDisplacement<CUDATileGameOfLife>();
}

// -- BitPack engine tests -----------------------------------------------------

TEST(CUDABitPack, Birth) {
    BitGrid bg(5, 5);
    bg.setCell(1, 1, true);
    bg.setCell(1, 2, true);
    bg.setCell(1, 3, true);
    CUDABitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2))
        << "dead cell with 3 neighbours should be born";
}

TEST(CUDABitPack, Underpopulation) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    CUDABitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2)) << "alone cell should die";
}

TEST(CUDABitPack, Overcrowding) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    bg.setCell(1, 1, true);
    bg.setCell(1, 2, true);
    bg.setCell(1, 3, true);
    bg.setCell(2, 1, true);
    CUDABitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2)) << "cell with 4 neighbours should die";
}

TEST(CUDABitPack, Survival) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    bg.setCell(1, 1, true);
    bg.setCell(1, 3, true);
    CUDABitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2)) << "cell with 2 neighbours should survive";
}

TEST(CUDABitPack, StillLifes) {
    BitGrid before(dataFile("still_lifes.txt"));
    size_t rows = before.getNumRows(), cols = before.getNumCols();
    BitGrid bg(dataFile("still_lifes.txt"));
    CUDABitPackGameOfLife engine(bg);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    BitGrid after = readBackBit(engine, rows, cols);
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            ASSERT_EQ(after.getCell(r, c), before.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

TEST(CUDABitPack, BlinkerPeriod2) {
    BitGrid g(5, 5);
    g.setCell(2, 1, true);
    g.setCell(2, 2, true);
    g.setCell(2, 3, true);

    BitGrid expected(5, 5);
    expected.setCell(2, 1, true);
    expected.setCell(2, 2, true);
    expected.setCell(2, 3, true);

    CUDABitPackGameOfLife engine(g);
    engine.takeStep();
    engine.takeStep();

    BitGrid after = readBackBit(engine, 5, 5);
    for (size_t r = 0; r < 5; r++)
        for (size_t c = 0; c < 5; c++)
            ASSERT_EQ(after.getCell(r, c), expected.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

TEST(CUDABitPack, AllDeadStaysDead) {
    BitGrid bg(10, 10);
    CUDABitPackGameOfLife engine(bg);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    BitGrid after = readBackBit(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}

// =============================================================================
// Direct kernel wrapper tests
// =============================================================================

// -- Byte kernel wrappers (simple + tile) ------------------------------------

class CUDAKernelByteTest : public ::testing::TestWithParam<void (*)(
                               const uint8_t *, uint8_t *, size_t, size_t)> {};

TEST_P(CUDAKernelByteTest, Birth) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[1 * cols + 1] = 1;
    in[1 * cols + 2] = 1;
    in[1 * cols + 3] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[2 * cols + 2], 1u)
        << "dead cell with 3 neighbours should be born";
}

TEST_P(CUDAKernelByteTest, Underpopulation) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[2 * cols + 2] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[2 * cols + 2], 0u) << "alone cell should die";
}

TEST_P(CUDAKernelByteTest, BlinkerPeriod2) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), mid(rows * cols, 0),
        out(rows * cols, 0);
    in[2 * cols + 1] = 1;
    in[2 * cols + 2] = 1;
    in[2 * cols + 3] = 1;
    GetParam()(in.data(), mid.data(), rows, cols);
    GetParam()(mid.data(), out.data(), rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
        ASSERT_EQ(out[i], in[i]) << "mismatch at linear index " << i;
}

TEST_P(CUDAKernelByteTest, CornerBirth) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[0 * cols + 1] = 1;
    in[1 * cols + 0] = 1;
    in[1 * cols + 1] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[0], 1u) << "corner (0,0) with 3 neighbours should be born";
}

TEST_P(CUDAKernelByteTest, NonTileAlignedGrid) {
    // 100x100: not a multiple of 128 (tile width)
    const size_t rows = 100, cols = 100;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    // Blinker at (50,50)
    in[50 * cols + 49] = 1;
    in[50 * cols + 50] = 1;
    in[50 * cols + 51] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[49 * cols + 50], 1u);
    EXPECT_EQ(out[50 * cols + 50], 1u);
    EXPECT_EQ(out[51 * cols + 50], 1u);
    EXPECT_EQ(out[50 * cols + 49], 0u);
    EXPECT_EQ(out[50 * cols + 51], 0u);
}

TEST_P(CUDAKernelByteTest, NonBlockAlignedGrid) {
    // 37x37: not a multiple of block dimensions
    const size_t rows = 37, cols = 37;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[18 * cols + 17] = 1;
    in[18 * cols + 18] = 1;
    in[18 * cols + 19] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[17 * cols + 18], 1u);
    EXPECT_EQ(out[18 * cols + 18], 1u);
    EXPECT_EQ(out[19 * cols + 18], 1u);
}

TEST_P(CUDAKernelByteTest, LargeGridStillLife) {
    // 512x512: block pattern repeated — verify stability
    const size_t rows = 512, cols = 512;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    // Place a 2x2 block (still life) at (100,100)
    in[100 * cols + 100] = 1;
    in[100 * cols + 101] = 1;
    in[101 * cols + 100] = 1;
    in[101 * cols + 101] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[100 * cols + 100], 1u);
    EXPECT_EQ(out[100 * cols + 101], 1u);
    EXPECT_EQ(out[101 * cols + 100], 1u);
    EXPECT_EQ(out[101 * cols + 101], 1u);
    // Count total alive — should be exactly 4
    size_t alive = 0;
    for (size_t i = 0; i < rows * cols; i++)
        alive += out[i];
    EXPECT_EQ(alive, 4u);
}

INSTANTIATE_TEST_SUITE_P(CUDAKernelByte, CUDAKernelByteTest,
                         ::testing::Values(cudaSimpleKernelStep,
                                           cudaTileKernelStep));

// -- BitPack kernel wrapper tests --------------------------------------------

TEST(CUDAKernelBitPack, Birth) {
    const size_t rows = 5, cols = 5;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), out(rows * stride, 0);
    in[1 * stride] |=
        (uint64_t(1) << 1) | (uint64_t(1) << 2) | (uint64_t(1) << 3);
    cudaBitPackKernelStep(in.data(), out.data(), rows, stride, cols);
    EXPECT_TRUE((out[2 * stride] >> 2) & 1)
        << "dead cell with 3 neighbours should be born";
}

TEST(CUDAKernelBitPack, BlinkerPeriod2) {
    const size_t rows = 5, cols = 5;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), mid(rows * stride, 0),
        out(rows * stride, 0);
    in[2 * stride] |=
        (uint64_t(1) << 1) | (uint64_t(1) << 2) | (uint64_t(1) << 3);
    cudaBitPackKernelStep(in.data(), mid.data(), rows, stride, cols);
    cudaBitPackKernelStep(mid.data(), out.data(), rows, stride, cols);
    for (size_t i = 0; i < rows * stride; i++)
        ASSERT_EQ(out[i], in[i]) << "mismatch at word " << i;
}

TEST(CUDAKernelBitPack, NonMultiple64Cols) {
    // 70 cols — last word has only 6 valid bits
    const size_t rows = 5, cols = 70;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), out(rows * stride, 0);
    // Blinker at row 2, cols 67-69
    in[2 * stride + 1] |= (uint64_t(1) << (67 - 64)) |
                          (uint64_t(1) << (68 - 64)) |
                          (uint64_t(1) << (69 - 64));
    cudaBitPackKernelStep(in.data(), out.data(), rows, stride, cols);
    // After 1 step: vertical at col 68
    EXPECT_TRUE((out[1 * stride + 1] >> (68 - 64)) & 1) << "row 1, col 68 born";
    EXPECT_TRUE((out[2 * stride + 1] >> (68 - 64)) & 1)
        << "row 2, col 68 survives";
    EXPECT_TRUE((out[3 * stride + 1] >> (68 - 64)) & 1) << "row 3, col 68 born";
    EXPECT_FALSE((out[2 * stride + 1] >> (67 - 64)) & 1)
        << "row 2, col 67 dies";
    EXPECT_FALSE((out[2 * stride + 1] >> (69 - 64)) & 1)
        << "row 2, col 69 dies";
}

TEST(CUDAKernelBitPack, LargeGridStillLife) {
    const size_t rows = 512, cols = 512;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), out(rows * stride, 0);
    // 2x2 block at (100,100)
    size_t w100 = 100 / 64, b100 = 100 % 64;
    size_t w101 = 101 / 64, b101 = 101 % 64;
    in[100 * stride + w100] |= uint64_t(1) << b100;
    in[100 * stride + w101] |= uint64_t(1) << b101;
    in[101 * stride + w100] |= uint64_t(1) << b100;
    in[101 * stride + w101] |= uint64_t(1) << b101;
    cudaBitPackKernelStep(in.data(), out.data(), rows, stride, cols);
    for (size_t i = 0; i < rows * stride; i++)
        ASSERT_EQ(out[i], in[i]) << "mismatch at word " << i;
}

#endif // GOL_CUDA
