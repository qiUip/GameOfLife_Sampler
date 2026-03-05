#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <omp.h>

#include "gol.h"
#include "gol_utils.h"

int main(int argc, char **argv) {
  int mpiRank, mpiSize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  Grid grid;
  SimParams params;

  if (mpiRank == 0) {
    if (!initSimulation(argc, argv, grid, params)) {
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    if (params.sleepTime >= 0) printStep(grid, "Generation:", 0);
  }

  if (mpiSize > 1) {
    mpiBroadcastSimInfo(params);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  omp_set_num_threads(params.numThreads);

  // Build local BitGrid — split across ranks if multi-rank
  BitGrid localBitGrid;
  if (mpiSize > 1) {
    if (mpiRank == 0) {
      BitGrid fullBitGrid(grid);
      mpiSplitGrid(localBitGrid, fullBitGrid, mpiSize);
    } else {
      mpiReceiveGrid(localBitGrid, 0);
    }
  } else {
    localBitGrid = BitGrid(grid);
  }

  auto game = std::make_unique<BitPackGameOfLife>(localBitGrid);

  // ── Single-rank path: pure OpenMP, no MPI communication ─────────────────
  if (mpiSize == 1) {
    auto t_start = std::chrono::high_resolution_clock::now();

    for (unsigned int step = 0; step < params.steps; ++step) {
      game->takeStep();
      if (params.sleepTime > 0 && step < params.steps - 1)
        printStep(game->getGrid(), "Generation:", step + 1, params.sleepTime);
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    const Grid finalGrid = game->getGrid();
    if (params.sleepTime >= 0) printStep(finalGrid, "Generation:", params.steps);
    if (!params.outfile.empty())
      finalGrid.writeToFile(params.outfile);
    std::cout << "Game completed in " << elapsed << " seconds\n";
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  // ── Multi-rank path: MPI domain decomposition + OpenMP ───────────────────
  auto t_start = std::chrono::high_resolution_clock::now();

  for (unsigned int step = 0; step < params.steps; ++step) {
    game->takeStep();
    exchangeBoundaryRows(*game, mpiRank, mpiSize);

    if (params.sleepTime > 0 && step < params.steps - 1) {
      if (mpiRank != 0) {
        assembleSend(*game, mpiRank, mpiSize);
      } else {
        Grid fullGrid = assembleFullGrid(*game, params.fullGridRows,
                                         params.fullGridColumns, mpiSize);
        printStep(fullGrid, "Generation:", step + 1, params.sleepTime);
      }
    }
  }

  double elapsed = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t_start).count();

  if (mpiRank != 0) {
    assembleSend(*game, mpiRank, mpiSize);
  } else {
    Grid finalGrid = assembleFullGrid(*game, params.fullGridRows,
                                      params.fullGridColumns, mpiSize);
    if (params.sleepTime >= 0) printStep(finalGrid, "Generation:", params.steps);
    if (!params.outfile.empty())
      finalGrid.writeToFile(params.outfile);
    std::cout << "Game completed in " << elapsed << " seconds\n";
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
