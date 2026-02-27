#include <chrono>
#include <iostream>
#include <mpi.h>
#include <omp.h>

#include "gol.h"
#include "gol_utils.h"

int main(int argc, char **argv) {
  // Initialize MPI
  int mpiRank, mpiSize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  // Parse CLI and create initial grid (rank 0 only)
  Grid grid;
  SimParams params;

  if (mpiRank == 0) {
    if (!initSimulation(argc, argv, grid, params)) {
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    if (params.sleepTime >= 0) printStep(grid, "Generation:", 0);
  }

  // ── Single-rank path: pure OpenMP, no MPI communication ─────────────────
  if (mpiSize == 1) {
    omp_set_num_threads(params.numThreads);
    //GameOfLife game(grid);
    BitGameOfLife game(grid);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (unsigned int step = 0; step < params.steps; ++step) {
      game.takeStep();
      if (params.sleepTime > 0 && step < params.steps - 1)
        printStep(game.getGrid(), "Generation:", step + 1, params.sleepTime);
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    const Grid finalGrid = game.getGrid();
    if (params.sleepTime >= 0) printStep(finalGrid, "Generation:", params.steps);
    if (!params.outfile.empty())
      finalGrid.writeToFile(params.outfile);
    //if (params.sleepTime >= 0) printStep(game.getGrid(), "Generation:", params.steps);
    //if (!params.outfile.empty())
    //  game.getGrid().writeToFile(params.outfile);
    std::cout << "Simulation completed in " << elapsed << " seconds\n";
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  // ── Multi-rank path: MPI domain decomposition + OpenMP ───────────────────
  mpiBroadcastSimInfo(params);
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpiRank == 0)
    mpiSplitGrid(grid, mpiSize);
  else
    mpiReceiveNewGrid(grid, 0);

  omp_set_num_threads(params.numThreads);

  const auto gridRows = grid.getNumRows();
  const auto gridColumns = grid.getNumColumns();

  GameOfLife game(grid);

  auto t_start = std::chrono::high_resolution_clock::now();

  for (unsigned int step = 0; step < params.steps; ++step) {
    game.takeStep();
    exchangeBoundaryRows(game, gridRows, gridColumns, mpiRank, mpiSize);

    if (params.sleepTime > 0 && step < params.steps - 1) {
      if (mpiRank != 0)
        assembleGridSend(game.getGrid(), gridRows, gridColumns, mpiRank, mpiSize);
      else {
        Grid assembled = assembleGrid(game.getGrid(), gridRows, gridColumns,
                                      params.fullGridRows, params.fullGridColumns,
                                      mpiRank, mpiSize);
        printStep(assembled, "Generation:", step + 1, params.sleepTime);
      }
    }
  }

  double elapsed = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t_start).count();

  if (mpiRank != 0) {
    assembleGridSend(game.getGrid(), gridRows, gridColumns, mpiRank, mpiSize);
  } else {
    grid = assembleGrid(game.getGrid(), gridRows, gridColumns,
                        params.fullGridRows, params.fullGridColumns,
                        mpiRank, mpiSize);
    if (params.sleepTime >= 0) printStep(grid, "Generation:", params.steps);
    if (!params.outfile.empty())
      grid.writeToFile(params.outfile);
    std::cout << "Simulation completed in " << elapsed << " seconds\n";
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
