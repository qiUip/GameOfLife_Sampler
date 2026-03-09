#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <omp.h>

#include "gol_mpi.h"
#include "gol_utils.h"

int main(int argc, char **argv) {
    int mpiRank, mpiSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    SimParams params;

    if (mpiRank == 0) {
        if (!initSimulation(argc, argv, params)) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (mpiSize > 1) {
        mpiBroadcastSimInfo(params);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    omp_set_num_threads(params.numThreads);

    std::unique_ptr<GameOfLife> game = createEngine(params, mpiRank, mpiSize);
    if (!game) {
        if (mpiRank == 0)
            std::cerr
                << "Error: engine not available (built without CUDA/HIP?)\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (mpiRank == 0) {
        printSimInfo(params);
        if (params.sleepTime >= 0)
            printStep(*game, "Generation:", 0);
    }

    // -- Single-rank path: pure OpenMP, no MPI communication -----------------
    if (mpiSize == 1) {
        auto t_start = std::chrono::high_resolution_clock::now();

        for (unsigned int step = 0; step < params.steps; ++step) {
            game->takeStep();
            if (params.sleepTime > 0 && step < params.steps - 1)
                printStep(*game, "Generation:", step + 1, params.sleepTime);
        }

        game->sync();
        double elapsed =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_start)
                .count();
        if (params.sleepTime >= 0)
            printStep(*game, "Generation:", params.steps);
        if (!params.outfile.empty())
            game->writeToFile(params.outfile);
        std::cout << "Game completed in " << elapsed << " seconds\n";
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    // -- Multi-rank path: MPI domain decomposition + OpenMP -------------------
    auto t_start = std::chrono::high_resolution_clock::now();

    for (unsigned int step = 0; step < params.steps; ++step) {
        game->takeStep();
        exchangeBoundaryRows(*game, mpiRank, mpiSize);
        game->commitBoundaries();

        if (params.sleepTime > 0 && step < params.steps - 1) {
            if (mpiRank != 0) {
                assembleSend(*game, mpiRank, mpiSize);
            } else {
                assembleOutput(*game, params.fullGridRows,
                               params.fullGridColumns, mpiSize,
                               params.sleepTime, step + 1);
            }
        }
    }

    game->sync();
    double elapsed = std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now() - t_start)
                         .count();

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
