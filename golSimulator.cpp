#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <omp.h>

#include "gol.h"
#include "gol_gpu.h"
#include "gol_utils.h"

template<typename Engine, typename GridType>
static std::unique_ptr<GameOfLife> setupGame(GridType &fullGrid, int mpiRank, int mpiSize) {
  GridType localGrid;
  if (mpiSize > 1) {
    if (mpiRank == 0)
      mpiSplitGrid(localGrid, fullGrid, mpiSize);
    else
      mpiReceiveGrid(localGrid, 0);
  } else {
    localGrid = std::move(fullGrid);
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
    if (params.sleepTime >= 0 && grid.getNumRows() > 0)
      printStep(grid, "Generation:", 0);
  }

  if (mpiSize > 1) {
    mpiBroadcastSimInfo(params);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  omp_set_num_threads(params.numThreads);

  std::unique_ptr<GameOfLife> game;
  switch (params.engine) {
  case ENGINE_SIMPLE:
    game = setupGame<SimpleGameOfLife>(grid, mpiRank, mpiSize);
    break;
  case ENGINE_SIMD:
    game = setupGame<SIMDGameOfLife>(grid, mpiRank, mpiSize);
    break;
  case ENGINE_BITPACK:
#if GOL_CUDA
  case ENGINE_CUDA_BITPACK:
#endif
#if GOL_HIP
  case ENGINE_HIP_BITPACK:
#endif
  {
    BitGrid bg;
    if (mpiRank == 0) {
      if (params.randomInit) {
        std::mt19937 rng(params.seed);
        bg = BitGrid(params.fullGridRows, params.fullGridColumns, params.alive, rng);
      } else {
        bg = BitGrid(grid);
      }
    }
#if GOL_CUDA
    if (params.engine == ENGINE_CUDA_BITPACK)
      game = setupGame<CUDABitPackGameOfLife>(bg, mpiRank, mpiSize);
    else
#endif
#if GOL_HIP
    if (params.engine == ENGINE_HIP_BITPACK)
      game = setupGame<HIPBitPackGameOfLife>(bg, mpiRank, mpiSize);
    else
#endif
      game = setupGame<BitPackGameOfLife>(bg, mpiRank, mpiSize);
    break;
  }
#if GOL_CUDA
  case ENGINE_CUDA_SIMPLE:
    game = setupGame<CUDASimpleGameOfLife>(grid, mpiRank, mpiSize);
    break;
  case ENGINE_CUDA_TILE:
    game = setupGame<CUDATileGameOfLife>(grid, mpiRank, mpiSize);
    break;
#endif
#if GOL_HIP
  case ENGINE_HIP_SIMPLE:
    game = setupGame<HIPSimpleGameOfLife>(grid, mpiRank, mpiSize);
    break;
  case ENGINE_HIP_TILE:
    game = setupGame<HIPTileGameOfLife>(grid, mpiRank, mpiSize);
    break;
#endif
  default:
    if (mpiRank == 0)
      std::cerr << "Error: engine not available (built without CUDA?)\n";
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // ── Single-rank path: pure OpenMP, no MPI communication ─────────────────
  if (mpiSize == 1) {
    auto t_start = std::chrono::high_resolution_clock::now();

    for (unsigned int step = 0; step < params.steps; ++step) {
      game->takeStep();
      if (params.sleepTime > 0 && step < params.steps - 1)
        printStep(*game, "Generation:", step + 1, params.sleepTime);
    }

    game->sync();
    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    if (params.sleepTime >= 0)
      printStep(*game, "Generation:", params.steps);
    if (!params.outfile.empty())
      game->writeToFile(params.outfile);
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
        assembleOutput(*game, params.fullGridRows, params.fullGridColumns,
                       mpiSize, params.sleepTime, step + 1);
      }
    }
  }

  game->sync();
  double elapsed = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t_start).count();

  if (mpiRank != 0) {
    assembleSend(*game, mpiRank, mpiSize);
  } else {
    assembleOutput(*game, params.fullGridRows, params.fullGridColumns,
                   mpiSize, params.sleepTime, params.steps, params.outfile);
    std::cout << "Game completed in " << elapsed << " seconds\n";
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
