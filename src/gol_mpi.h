#ifndef GOL_MPI_H
#define GOL_MPI_H

#include "gol_engine.h"
#include "gol_utils.h"
#include <string>

// MPI utilities for domain decomposition and boundary exchange
void mpiBroadcastSimInfo(SimParams &params);
void exchangeBoundaryRows(GameOfLife &game, int mpiRank, int mpiSize);
void assembleSend(GameOfLife &game, int mpiRank, int mpiSize);
void assembleOutput(GameOfLife &game, size_t fullRows, size_t fullCols,
                    int mpiSize, float sleepTime, int step,
                    const std::string &outfile = "");

template <typename GridType> void mpiReceiveGrid(GridType &grid, int rank);
template <typename GridType>
void mpiSplitGrid(GridType &localGrid, const GridType &fullGrid, int mpiSize);

#endif // GOL_MPI_H
