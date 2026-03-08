#include <gtest/gtest.h>

#include "gol_utils.h"

#include <cstring>
#include <vector>

// Helper to build argc/argv from a string list
struct Args {
    std::vector<std::string> strs;
    std::vector<char *> ptrs;

    Args(std::initializer_list<const char *> args) {
        for (auto *a : args)
            strs.emplace_back(a);
        for (auto &s : strs)
            ptrs.push_back(s.data());
    }

    int argc() {
        return static_cast<int>(ptrs.size());
    }
    char **argv() {
        return ptrs.data();
    }
};

TEST(Utils, ParseValidArgs) {
    Args a{"golSimulator", "-r",   "100,50,200", "-g", "10",
           "-e",           "simd", "-s",         "42"};
    // Reset getopt state
    optind = 1;
    SimParams p;
    ASSERT_TRUE(initSimulation(a.argc(), a.argv(), p));
    EXPECT_EQ(p.fullGridRows, 100u);
    EXPECT_EQ(p.fullGridColumns, 50u);
    EXPECT_EQ(p.alive, 200u);
    EXPECT_EQ(p.steps, 10u);
    EXPECT_EQ(p.engine, ENGINE_SIMD);
    EXPECT_EQ(p.seed, 42u);
    EXPECT_TRUE(p.randomInit);
}

TEST(Utils, UnknownEngineReturnsFalse) {
    Args a{"golSimulator", "-r", "10,10,5", "-e", "nonexistent"};
    optind = 1;
    SimParams p;
    EXPECT_FALSE(initSimulation(a.argc(), a.argv(), p));
}

TEST(Utils, MissingArgsReturnsFalse) {
    Args a{"golSimulator"};
    optind = 1;
    SimParams p;
    EXPECT_FALSE(initSimulation(a.argc(), a.argv(), p));
}
