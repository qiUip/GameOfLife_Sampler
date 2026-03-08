#include "gol_utils.h"
#include "gol_gpu.h"
#include "gol_mpi.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

#include <getopt.h>
#include <unistd.h>

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::vector<size_t> parseArgs(const std::string &args) {
    std::vector<size_t> values;
    std::istringstream ss(args);
    std::string token;
    while (std::getline(ss, token, ',')) {
        values.push_back(std::stoul(token));
    }
    return values;
}

static const std::pair<const char *, Engine> engineNames[] = {
    {"simple", ENGINE_SIMPLE},
    {"simd", ENGINE_SIMD},
    {"bitpack", ENGINE_BITPACK},
    {"cuda-simple", ENGINE_CUDA_SIMPLE},
    {"cuda-tile", ENGINE_CUDA_TILE},
    {"cuda-bitpack", ENGINE_CUDA_BITPACK},
    {"hip-simple", ENGINE_HIP_SIMPLE},
    {"hip-tile", ENGINE_HIP_TILE},
    {"hip-bitpack", ENGINE_HIP_BITPACK},
};

// ── Print functions ─────────────────────────────────────────────────────────

void printHelp() {
    std::cerr
        << "Usage: golSimulator [OPTIONS]\n"
        << "Options:\n"
        << "  -f, --file <filename>.txt    Initialize from file\n"
        << "  -r, --random <rows,cols,alive>  Initialize random grid\n"
        << "  -s, --seed <int>             Seed for random initialization\n"
        << "  -g, --generations <int>      Number of generations\n"
        << "  -p, --print <float>          Print each step (delay in seconds)\n"
        << "  -o, --output <filename>.txt  Output file name\n"
        << "  -n, --numthreads <int>       Number of OpenMP threads\n"
        << "  -e, --engine <name>          Engine: ";
    for (size_t i = 0; i < std::size(engineNames); i++) {
        if (i > 0)
            std::cerr << ", ";
        std::cerr << engineNames[i].first;
    }
    std::cerr << "\n  -h, --help                   Print this help message\n";
}

void printLine(size_t length) {
    for (size_t i = 0; i < length; i++)
        std::cout << "==";
    std::cout << "\n";
}

void printSimInfo(const SimParams &params) {
    std::cout << "Initializing GameOfLife with grid size ("
              << params.fullGridRows << ", " << params.fullGridColumns
              << ") for " << params.steps << " generations\n";
}

bool initSimulation(int argc, char **argv, SimParams &params) {
    const char *const short_opts = "f:r:s:g:n:p:o:e:h";
    const option long_opts[]     = {
        {"file", required_argument, nullptr, 'f'},
        {"random", required_argument, nullptr, 'r'},
        {"seed", required_argument, nullptr, 's'},
        {"generations", required_argument, nullptr, 'g'},
        {"numthreads", required_argument, nullptr, 'n'},
        {"print", optional_argument, nullptr, 'p'},
        {"output", required_argument, nullptr, 'o'},
        {"engine", required_argument, nullptr, 'e'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, no_argument, nullptr, 0}};

    std::string randomArgs;
    bool seedProvided = false;

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) !=
           -1) {
        switch (opt) {
            case 'f':
                params.filename = optarg;
                break;
            case 'r':
                randomArgs = optarg;
                break;
            case 's':
                params.seed  = std::stoi(optarg);
                seedProvided = true;
                break;
            case 'g':
                params.steps = std::stoi(optarg);
                break;
            case 'p':
                params.sleepTime = (optarg ? std::stof(optarg) : 0.1f);
                break;
            case 'o':
                params.outfile = optarg;
                break;
            case 'n':
                params.numThreads = std::stoi(optarg);
                break;
            case 'e': {
                std::string eng = optarg;
                bool found      = false;
                for (const auto &[name, id] : engineNames)
                    if (eng == name) {
                        params.engine = id;
                        found         = true;
                        break;
                    }
                if (!found) {
                    std::cerr << "Error: unknown engine '" << eng << "'\n";
                    printHelp();
                    return false;
                }
                break;
            }
            case 'h':
                printHelp();
                return false;
            default:
                printHelp();
                return false;
        }
    }

    if (!params.filename.empty()) {
        // Dimensions determined when the grid is constructed in createEngine
    } else if (!randomArgs.empty()) {
        auto values = parseArgs(randomArgs);
        if (values.size() != 3) {
            std::cerr << "Error: -r expects rows,cols,alive\n";
            return false;
        }
        params.fullGridRows    = values[0];
        params.fullGridColumns = values[1];
        params.alive           = values[2];
        params.randomInit      = true;
        if (!seedProvided)
            params.seed = std::random_device()();
    } else {
        std::cerr << "Error: No initialisation option provided.\n";
        printHelp();
        return false;
    }

    return true;
}

// ── Engine factory ──────────────────────────────────────────────────────────

template <typename GridType>
static GridType makeGrid(const SimParams &params) {
    if (params.randomInit) {
        std::mt19937 rng(params.seed);
        return GridType(params.fullGridRows, params.fullGridColumns,
                        params.alive, rng);
    }
    return GridType(params.filename);
}

template <typename EngineType, typename GridType>
static std::unique_ptr<GameOfLife> setupEngine(SimParams &params, int mpiRank,
                                               int mpiSize) {
    GridType grid;
    if (mpiRank == 0) {
        grid = makeGrid<GridType>(params);
        params.fullGridRows    = grid.getNumRows();
        params.fullGridColumns = grid.getNumCols();
    }
    if (mpiSize > 1) {
        GridType localGrid;
        if (mpiRank == 0)
            mpiSplitGrid(localGrid, grid, mpiSize);
        else
            mpiReceiveGrid(localGrid, 0);
        return std::make_unique<EngineType>(localGrid);
    }
    return std::make_unique<EngineType>(grid);
}

std::unique_ptr<GameOfLife> createEngine(SimParams &params, int mpiRank,
                                         int mpiSize) {
    switch (params.engine) {
        case ENGINE_SIMPLE:
            return setupEngine<SimpleGameOfLife, Grid>(params, mpiRank, mpiSize);
        case ENGINE_SIMD:
            return setupEngine<SIMDGameOfLife, Grid>(params, mpiRank, mpiSize);
        case ENGINE_BITPACK:
            return setupEngine<BitPackGameOfLife, BitGrid>(params, mpiRank, mpiSize);
#if GOL_CUDA
        case ENGINE_CUDA_SIMPLE:
            return setupEngine<CUDASimpleGameOfLife, Grid>(params, mpiRank, mpiSize);
        case ENGINE_CUDA_TILE:
            return setupEngine<CUDATileGameOfLife, Grid>(params, mpiRank, mpiSize);
        case ENGINE_CUDA_BITPACK:
            return setupEngine<CUDABitPackGameOfLife, BitGrid>(params, mpiRank, mpiSize);
#endif
#if GOL_HIP
        case ENGINE_HIP_SIMPLE:
            return setupEngine<HIPSimpleGameOfLife, Grid>(params, mpiRank, mpiSize);
        case ENGINE_HIP_TILE:
            return setupEngine<HIPTileGameOfLife, Grid>(params, mpiRank, mpiSize);
        case ENGINE_HIP_BITPACK:
            return setupEngine<HIPBitPackGameOfLife, BitGrid>(params, mpiRank, mpiSize);
#endif
        default:
            return nullptr;
    }
}

// ── Print utilities ─────────────────────────────────────────────────────────

template <typename T>
void printStep(const T &grid, const std::string &label, int value,
               float sleepTime) {
    std::cout << label << " " << value << "\n";
    printLine(grid.getNumCols());
    grid.printGrid();
    printLine(grid.getNumCols());
    if (sleepTime > 0) {
        std::cout.flush();
        std::cout << "\033[" << grid.getNumRows() + 3 << "A";
        usleep(static_cast<unsigned int>(sleepTime * 1.0e6));
    } else {
        std::cout.flush();
    }
}

template void printStep(const Grid &, const std::string &, int, float);
template void printStep(const BitGrid &, const std::string &, int, float);
template void printStep(const GameOfLife &, const std::string &, int, float);
