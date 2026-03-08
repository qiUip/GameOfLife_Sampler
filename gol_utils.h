#ifndef GOL_UTILS_H
#define GOL_UTILS_H

#include "gol.h"
#include <memory>
#include <string>

enum Engine : int {
    ENGINE_SIMD = 0,
    ENGINE_BITPACK,
    ENGINE_CUDA_TILE,
    ENGINE_CUDA_SIMPLE,
    ENGINE_CUDA_BITPACK,
    ENGINE_SIMPLE,
    ENGINE_HIP_SIMPLE,
    ENGINE_HIP_TILE,
    ENGINE_HIP_BITPACK
};

struct SimParams {
    unsigned int steps = 1;
    unsigned int seed  = 0;
    float sleepTime    = 0;
    std::string outfile;
    std::string filename;
    int numThreads         = 1;
    Engine engine          = ENGINE_SIMD;
    size_t fullGridRows    = 0;
    size_t fullGridColumns = 0;
    unsigned int alive     = 0;
    bool randomInit        = false;
};

bool initSimulation(int argc, char **argv, SimParams &params);
std::unique_ptr<GameOfLife> createEngine(SimParams &params, int mpiRank,
                                         int mpiSize);

void printHelp();
void printLine(size_t length);
void printSimInfo(const SimParams &params);

template <typename T>
void printStep(const T &grid, const std::string &label, int value,
               float sleepTime = 0);

#endif // GOL_UTILS_H
