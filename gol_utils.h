#ifndef GOL_UTILS_H
#define GOL_UTILS_H

#include "gol.h"
#include <string>
#include <vector>

enum Engine : int { ENGINE_SIMD = 0, ENGINE_BITPACK, ENGINE_CUDA, ENGINE_CUDA_COLBATCH, ENGINE_CUDA_BITPACK, ENGINE_SIMPLE };

struct SimParams {
  unsigned int steps = 1;
  unsigned int seed = 0;
  float sleepTime = 0;
  std::string outfile;
  int numThreads = 1;
  int engine = ENGINE_SIMD;
  size_t fullGridRows = 0;
  size_t fullGridColumns = 0;
  unsigned int alive = 0;
  bool randomInit = false;
};

bool initSimulation(int argc, char **argv, Grid &grid, SimParams &params);

void printHelp();
void printLine(size_t length);
void printStep(const Grid &grid, const std::string &label, int value,
               float sleepTime = 0);

// MPI utilities
void mpiBroadcastSimInfo(SimParams &params);
void exchangeBoundaryRows(GameOfLife &game, int mpiRank, int mpiSize);
void assembleSend(GameOfLife &game, int mpiRank, int mpiSize);
Grid assembleFullGrid(GameOfLife &game, size_t fullRows, size_t fullCols,
                      int mpiSize);

template<typename GridType> void mpiReceiveGrid(GridType &grid, int rank);
template<typename GridType> void mpiSplitGrid(GridType &localGrid,
                                               const GridType &fullGrid,
                                               int mpiSize);

#endif // GOL_UTILS_H
