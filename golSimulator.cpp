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
    printStep(grid, "Generation:", 0);
  }

  // Broadcast simulation parameters to all ranks
  mpiBroadcastSimInfo(params);
  MPI_Barrier(MPI_COMM_WORLD);

  // Distribute grid using domain decomposition
  if (mpiRank == 0) {
    mpiSplitGrid(grid, mpiSize);
  } else {
    mpiReceiveNewGrid(grid, 0);
  }

  omp_set_num_threads(params.numThreads);

  const auto gridRows = grid.getNumRows();
  const auto gridColumns = grid.getNumColumns();

  GameOfLife game(grid);

  // Start timing
  auto t_start = std::chrono::high_resolution_clock::now();

  // Main simulation loop
  for (unsigned int step = 0; step < params.steps; ++step) {
    game.takeStep();
    exchangeBoundaryRows(game, gridRows, gridColumns, mpiRank, mpiSize);

    // Print intermediate steps if animation enabled
    if (params.sleepTime > 0 && step < params.steps - 1) {
      if (mpiRank != 0) {
        assembleGridSend(game.getGrid(), gridRows, gridColumns, mpiRank, mpiSize);
      } else {
        Grid assembled = assembleGrid(game.getGrid(), gridRows, gridColumns,
                                      params.fullGridRows, params.fullGridColumns,
                                      mpiRank, mpiSize);
        printStep(assembled, "Generation:", step + 1, params.sleepTime);
      }
    }
  }

  // End timing
  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  // Assemble final result at rank 0
  if (mpiRank != 0) {
    assembleGridSend(game.getGrid(), gridRows, gridColumns, mpiRank, mpiSize);
  } else {
    grid = assembleGrid(game.getGrid(), gridRows, gridColumns,
                        params.fullGridRows, params.fullGridColumns,
                        mpiRank, mpiSize);
    printStep(grid, "Generation:", params.steps);

    if (!params.outfile.empty()) {
      grid.writeToFile(params.outfile);
    }
    std::cout << "Simulation completed in " << elapsed << " seconds\n";
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
