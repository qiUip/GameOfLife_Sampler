#ifndef GOL_UTILS_H
#define GOL_UTILS_H

#include "gol.h"
#include <string>
#include <vector>

// Simulation parameters structure
struct SimParams {
  unsigned int steps = 1;
  unsigned int seed = 0;
  float sleepTime = 0;
  std::string outfile;
  int numThreads = 1;
  size_t fullGridRows = 0;
  size_t fullGridColumns = 0;
};

// Initialize simulation: parse CLI arguments and create initial grid
// Returns true on success, false on failure (help requested or error)
bool initSimulation(int argc, char **argv, Grid &grid, SimParams &params);

// Print functions
void printHelp();
void printLine(size_t length);
void printStep(const Grid &grid, const std::string &label, int value,
               float sleepTime=0);

// MPI utility functions
void mpiBroadcastSimInfo(SimParams &params);
void mpiSendNewGrid(const Grid &grid, int rank);
void mpiReceiveNewGrid(Grid &grid, int rank);
void mpiSplitGrid(Grid &fullGrid, int mpiSize);
void exchangeBoundaryRows(GameOfLife &game, size_t gridRows, size_t gridColumns,
                          int mpiRank, int mpiSize);
void assembleGridSend(const Grid &grid, size_t gridRows, size_t gridColumns,
                      int mpiRank, int mpiSize);
Grid assembleGrid(const Grid &grid, size_t gridRows, size_t gridColumns,
                  size_t fullGridRows, size_t fullGridColumns, int mpiRank,
                  int mpiSize);

#endif // GOL_UTILS_H
