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
bool initSimulation(int argc, char **argv, Grid &grid, SimParams &params);

// Print functions
void printHelp();
void printLine(size_t length);
void printStep(const Grid &grid, const std::string &label, int value,
               float sleepTime = 0);

// ── Unified MPI utilities (use virtual base class) ──────────────────────────
void mpiBroadcastSimInfo(SimParams &params);
void exchangeBoundaryRows(GameOfLifeBase &game, int mpiRank, int mpiSize);
void assembleSend(GameOfLifeBase &game, int mpiRank, int mpiSize);
Grid assembleFullGrid(GameOfLifeBase &game, size_t fullRows, size_t fullCols,
                      int mpiSize);

// ── Templated grid-level MPI (for pre-game-construction operations) ─────────
template<typename GridType> void mpiSendGrid(const GridType &grid, int rank);
template<typename GridType> void mpiReceiveGrid(GridType &grid, int rank);
template<typename GridType> void mpiSplitGrid(GridType &localGrid,
                                               const GridType &fullGrid,
                                               int mpiSize);

#endif // GOL_UTILS_H
