#include <chrono>
#include <iostream>
#include <memory>
#include <type_traits>
#include <mpi.h>
#include <omp.h>

#include "gol.h"
#include "gol_utils.h"

template<typename LocalGrid, typename Engine>
static std::unique_ptr<GameOfLife> setupGame(Grid &grid, int mpiRank, int mpiSize) {
  LocalGrid localGrid;
  if (mpiSize > 1) {
    if (mpiRank == 0) {
      if constexpr (std::is_same_v<LocalGrid, Grid>) {
        mpiSplitGrid(localGrid, grid, mpiSize);
      } else {
        LocalGrid fullGrid(grid);
        mpiSplitGrid(localGrid, fullGrid, mpiSize);
      }
    } else {
      mpiReceiveGrid(localGrid, 0);
    }
  } else {
    if constexpr (std::is_same_v<LocalGrid, Grid>)
      localGrid = std::move(grid);
    else
      localGrid = LocalGrid(grid);
  }
  return std::make_unique<Engine>(localGrid);
}

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

  // Change these two type args to switch engine:
  // auto game = setupGame<BitGrid, BitPackGameOfLife>(grid, mpiRank, mpiSize);
#if GOL_CUDA
  // auto game = setupGame<Grid, CUDAGameOfLife>(grid, mpiRank, mpiSize);
  auto game = setupGame<BitGrid, CUDABitPackGameOfLife>(grid, mpiRank, mpiSize);
#else
  auto game = setupGame<Grid, SIMDGameOfLife>(grid, mpiRank, mpiSize);
#endif

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
    game->commitBoundaries();

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
