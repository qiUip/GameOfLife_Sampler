#ifndef GOL_UTILS_H
#define GOL_UTILS_H

#include "gol.h"
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

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
    int numThreads         = 1;
    int engine             = ENGINE_SIMD;
    size_t fullGridRows    = 0;
    size_t fullGridColumns = 0;
    unsigned int alive     = 0;
    bool randomInit        = false;
};

bool initSimulation(int argc, char **argv, Grid &grid, SimParams &params);

void printHelp();
void printLine(size_t length);

template <typename T>
void printStep(const T &grid, const std::string &label, int value,
               float sleepTime = 0) {
    std::cout << label << " " << value << "\n";
    printLine(grid.getNumCols());
    grid.printGrid();
    printLine(grid.getNumCols());
    if (sleepTime > 0) {
        std::cout.flush();
        std::cout << "\033[" << grid.getNumRows() + 3 << "A";
        usleep(static_cast<unsigned int>(sleepTime * 1.0e6));
    } else {
        std::cout.flush();
    }
}

// MPI utilities
void mpiBroadcastSimInfo(SimParams &params);
void exchangeBoundaryRows(GameOfLife &game, int mpiRank, int mpiSize);
void assembleSend(GameOfLife &game, int mpiRank, int mpiSize);
void assembleOutput(GameOfLife &game, size_t fullRows, size_t fullCols,
                    int mpiSize, float sleepTime, int step,
                    const std::string &outfile = "");

template <typename GridType> void mpiReceiveGrid(GridType &grid, int rank);
template <typename GridType>
void mpiSplitGrid(GridType &localGrid, const GridType &fullGrid, int mpiSize);

#endif // GOL_UTILS_H
