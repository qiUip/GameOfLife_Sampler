#if GOL_HIP

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

TEST(HIPRule, Birth_Simple) {
    testBirth<HIPSimpleGameOfLife>();
}
TEST(HIPRule, Birth_Tile) {
    testBirth<HIPTileGameOfLife>();
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

TEST(HIPRule, Survival2_Simple) {
    testSurvival2<HIPSimpleGameOfLife>();
}
TEST(HIPRule, Survival2_Tile) {
    testSurvival2<HIPTileGameOfLife>();
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

TEST(HIPRule, Survival3_Simple) {
    testSurvival3<HIPSimpleGameOfLife>();
}
TEST(HIPRule, Survival3_Tile) {
    testSurvival3<HIPTileGameOfLife>();
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

TEST(HIPRule, Underpop0_Simple) {
    testUnderpopulation0<HIPSimpleGameOfLife>();
}
TEST(HIPRule, Underpop0_Tile) {
    testUnderpopulation0<HIPTileGameOfLife>();
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

TEST(HIPRule, Underpop1_Simple) {
    testUnderpopulation1<HIPSimpleGameOfLife>();
}
TEST(HIPRule, Underpop1_Tile) {
    testUnderpopulation1<HIPTileGameOfLife>();
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

TEST(HIPRule, Overcrowding_Simple) {
    testOvercrowding4<HIPSimpleGameOfLife>();
}
TEST(HIPRule, Overcrowding_Tile) {
    testOvercrowding4<HIPTileGameOfLife>();
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

TEST(HIPRule, DeadStaysDead_Simple) {
    testDeadStaysDead<HIPSimpleGameOfLife>();
}
TEST(HIPRule, DeadStaysDead_Tile) {
    testDeadStaysDead<HIPTileGameOfLife>();
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

TEST(HIPRule, DeadWith4_Simple) {
    testDeadWith4<HIPSimpleGameOfLife>();
}
TEST(HIPRule, DeadWith4_Tile) {
    testDeadWith4<HIPTileGameOfLife>();
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

TEST(HIPBoundary, CornerTopLeft_Simple) {
    testCornerTopLeft<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, CornerTopLeft_Tile) {
    testCornerTopLeft<HIPTileGameOfLife>();
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

TEST(HIPBoundary, CornerTopRight_Simple) {
    testCornerTopRight<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, CornerTopRight_Tile) {
    testCornerTopRight<HIPTileGameOfLife>();
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

TEST(HIPBoundary, CornerBottomLeft_Simple) {
    testCornerBottomLeft<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, CornerBottomLeft_Tile) {
    testCornerBottomLeft<HIPTileGameOfLife>();
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

TEST(HIPBoundary, CornerBottomRight_Simple) {
    testCornerBottomRight<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, CornerBottomRight_Tile) {
    testCornerBottomRight<HIPTileGameOfLife>();
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

TEST(HIPBoundary, TopEdge_Simple) {
    testTopEdge<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, TopEdge_Tile) {
    testTopEdge<HIPTileGameOfLife>();
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

TEST(HIPBoundary, LeftEdge_Simple) {
    testLeftEdge<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, LeftEdge_Tile) {
    testLeftEdge<HIPTileGameOfLife>();
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

TEST(HIPBoundary, RightEdge_Simple) {
    testRightEdge<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, RightEdge_Tile) {
    testRightEdge<HIPTileGameOfLife>();
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

TEST(HIPBoundary, BottomEdge_Simple) {
    testBottomEdge<HIPSimpleGameOfLife>();
}
TEST(HIPBoundary, BottomEdge_Tile) {
    testBottomEdge<HIPTileGameOfLife>();
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

TEST(HIPPattern, StillLifes_Simple) {
    testStillLifesStable<HIPSimpleGameOfLife>();
}
TEST(HIPPattern, StillLifes_Tile) {
    testStillLifesStable<HIPTileGameOfLife>();
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

TEST(HIPPattern, Blinker_Simple) {
    testBlinkerPeriod2<HIPSimpleGameOfLife>();
}
TEST(HIPPattern, Blinker_Tile) {
    testBlinkerPeriod2<HIPTileGameOfLife>();
}

template <typename EngineT> void testAllDeadStaysDead() {
    Grid g(10, 10);
    EngineT engine(g);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    Grid after = readBack(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}

TEST(HIPPattern, AllDead_Simple) {
    testAllDeadStaysDead<HIPSimpleGameOfLife>();
}
TEST(HIPPattern, AllDead_Tile) {
    testAllDeadStaysDead<HIPTileGameOfLife>();
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

TEST(HIPPattern, Glider_Simple) {
    testGliderDisplacement<HIPSimpleGameOfLife>();
}
TEST(HIPPattern, Glider_Tile) {
    testGliderDisplacement<HIPTileGameOfLife>();
}

// -- BitPack engine tests -----------------------------------------------------

TEST(HIPBitPack, Birth) {
    BitGrid bg(5, 5);
    bg.setCell(1, 1, true);
    bg.setCell(1, 2, true);
    bg.setCell(1, 3, true);
    HIPBitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2))
        << "dead cell with 3 neighbours should be born";
}

TEST(HIPBitPack, Underpopulation) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    HIPBitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2)) << "alone cell should die";
}

TEST(HIPBitPack, Overcrowding) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    bg.setCell(1, 1, true);
    bg.setCell(1, 2, true);
    bg.setCell(1, 3, true);
    bg.setCell(2, 1, true);
    HIPBitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_FALSE(after.getCell(2, 2)) << "cell with 4 neighbours should die";
}

TEST(HIPBitPack, Survival) {
    BitGrid bg(5, 5);
    bg.setCell(2, 2, true);
    bg.setCell(1, 1, true);
    bg.setCell(1, 3, true);
    HIPBitPackGameOfLife engine(bg);
    engine.takeStep();
    BitGrid after = readBackBit(engine, 5, 5);
    EXPECT_TRUE(after.getCell(2, 2)) << "cell with 2 neighbours should survive";
}

TEST(HIPBitPack, StillLifes) {
    BitGrid before(dataFile("still_lifes.txt"));
    size_t rows = before.getNumRows(), cols = before.getNumCols();
    BitGrid bg(dataFile("still_lifes.txt"));
    HIPBitPackGameOfLife engine(bg);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    BitGrid after = readBackBit(engine, rows, cols);
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            ASSERT_EQ(after.getCell(r, c), before.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

TEST(HIPBitPack, BlinkerPeriod2) {
    BitGrid g(5, 5);
    g.setCell(2, 1, true);
    g.setCell(2, 2, true);
    g.setCell(2, 3, true);

    BitGrid expected(5, 5);
    expected.setCell(2, 1, true);
    expected.setCell(2, 2, true);
    expected.setCell(2, 3, true);

    HIPBitPackGameOfLife engine(g);
    engine.takeStep();
    engine.takeStep();

    BitGrid after = readBackBit(engine, 5, 5);
    for (size_t r = 0; r < 5; r++)
        for (size_t c = 0; c < 5; c++)
            ASSERT_EQ(after.getCell(r, c), expected.getCell(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

TEST(HIPBitPack, AllDeadStaysDead) {
    BitGrid bg(10, 10);
    HIPBitPackGameOfLife engine(bg);
    for (int i = 0; i < 10; i++)
        engine.takeStep();
    BitGrid after = readBackBit(engine, 10, 10);
    EXPECT_EQ(after.aliveCells(), 0u);
}

// =============================================================================
// Direct kernel wrapper tests
// =============================================================================

// -- Byte kernel wrappers (simple + tile) ------------------------------------

class HIPKernelByteTest : public ::testing::TestWithParam<void (*)(
                              const uint8_t *, uint8_t *, size_t, size_t)> {};

TEST_P(HIPKernelByteTest, Birth) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[1 * cols + 1] = 1;
    in[1 * cols + 2] = 1;
    in[1 * cols + 3] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[2 * cols + 2], 1u)
        << "dead cell with 3 neighbours should be born";
}

TEST_P(HIPKernelByteTest, Underpopulation) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[2 * cols + 2] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[2 * cols + 2], 0u) << "alone cell should die";
}

TEST_P(HIPKernelByteTest, BlinkerPeriod2) {
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

TEST_P(HIPKernelByteTest, CornerBirth) {
    const size_t rows = 5, cols = 5;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[0 * cols + 1] = 1;
    in[1 * cols + 0] = 1;
    in[1 * cols + 1] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[0], 1u) << "corner (0,0) with 3 neighbours should be born";
}

TEST_P(HIPKernelByteTest, NonTileAlignedGrid) {
    // 100x100: not a multiple of 256 (HIP tile width)
    const size_t rows = 100, cols = 100;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
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

TEST_P(HIPKernelByteTest, NonBlockAlignedGrid) {
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

TEST_P(HIPKernelByteTest, LargeGridStillLife) {
    const size_t rows = 512, cols = 512;
    std::vector<uint8_t> in(rows * cols, 0), out(rows * cols, 0);
    in[100 * cols + 100] = 1;
    in[100 * cols + 101] = 1;
    in[101 * cols + 100] = 1;
    in[101 * cols + 101] = 1;
    GetParam()(in.data(), out.data(), rows, cols);
    EXPECT_EQ(out[100 * cols + 100], 1u);
    EXPECT_EQ(out[100 * cols + 101], 1u);
    EXPECT_EQ(out[101 * cols + 100], 1u);
    EXPECT_EQ(out[101 * cols + 101], 1u);
    size_t alive = 0;
    for (size_t i = 0; i < rows * cols; i++)
        alive += out[i];
    EXPECT_EQ(alive, 4u);
}

INSTANTIATE_TEST_SUITE_P(HIPKernelByte, HIPKernelByteTest,
                         ::testing::Values(hipSimpleKernelStep,
                                           hipTileKernelStep));

// -- BitPack kernel wrapper tests --------------------------------------------

TEST(HIPKernelBitPack, Birth) {
    const size_t rows = 5, cols = 5;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), out(rows * stride, 0);
    in[1 * stride] |=
        (uint64_t(1) << 1) | (uint64_t(1) << 2) | (uint64_t(1) << 3);
    hipBitPackKernelStep(in.data(), out.data(), rows, stride, cols);
    EXPECT_TRUE((out[2 * stride] >> 2) & 1)
        << "dead cell with 3 neighbours should be born";
}

TEST(HIPKernelBitPack, BlinkerPeriod2) {
    const size_t rows = 5, cols = 5;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), mid(rows * stride, 0),
        out(rows * stride, 0);
    in[2 * stride] |=
        (uint64_t(1) << 1) | (uint64_t(1) << 2) | (uint64_t(1) << 3);
    hipBitPackKernelStep(in.data(), mid.data(), rows, stride, cols);
    hipBitPackKernelStep(mid.data(), out.data(), rows, stride, cols);
    for (size_t i = 0; i < rows * stride; i++)
        ASSERT_EQ(out[i], in[i]) << "mismatch at word " << i;
}

TEST(HIPKernelBitPack, NonMultiple64Cols) {
    const size_t rows = 5, cols = 70;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), out(rows * stride, 0);
    in[2 * stride + 1] |= (uint64_t(1) << (67 - 64)) |
                          (uint64_t(1) << (68 - 64)) |
                          (uint64_t(1) << (69 - 64));
    hipBitPackKernelStep(in.data(), out.data(), rows, stride, cols);
    EXPECT_TRUE((out[1 * stride + 1] >> (68 - 64)) & 1) << "row 1, col 68 born";
    EXPECT_TRUE((out[2 * stride + 1] >> (68 - 64)) & 1)
        << "row 2, col 68 survives";
    EXPECT_TRUE((out[3 * stride + 1] >> (68 - 64)) & 1) << "row 3, col 68 born";
    EXPECT_FALSE((out[2 * stride + 1] >> (67 - 64)) & 1)
        << "row 2, col 67 dies";
    EXPECT_FALSE((out[2 * stride + 1] >> (69 - 64)) & 1)
        << "row 2, col 69 dies";
}

TEST(HIPKernelBitPack, LargeGridStillLife) {
    const size_t rows = 512, cols = 512;
    size_t stride = (cols + 63) / 64;
    std::vector<uint64_t> in(rows * stride, 0), out(rows * stride, 0);
    size_t w100 = 100 / 64, b100 = 100 % 64;
    size_t w101 = 101 / 64, b101 = 101 % 64;
    in[100 * stride + w100] |= uint64_t(1) << b100;
    in[100 * stride + w101] |= uint64_t(1) << b101;
    in[101 * stride + w100] |= uint64_t(1) << b100;
    in[101 * stride + w101] |= uint64_t(1) << b101;
    hipBitPackKernelStep(in.data(), out.data(), rows, stride, cols);
    for (size_t i = 0; i < rows * stride; i++)
        ASSERT_EQ(out[i], in[i]) << "mismatch at word " << i;
}

#endif // GOL_HIP
