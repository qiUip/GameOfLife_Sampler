#include "gol_utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <random>
#include <sstream>

#include <getopt.h>
#include <mpi.h>
#include <unistd.h>

// ── MPI datatype trait ──────────────────────────────────────────────────────

template<typename T> static MPI_Datatype mpiDatatype();
template<> MPI_Datatype mpiDatatype<uint8_t>()  { return MPI_UINT8_T; }
template<> MPI_Datatype mpiDatatype<uint64_t>() { return MPI_UINT64_T; }

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

static std::vector<std::pair<size_t, size_t>> getRowRanges(int numRanks,
                                                           size_t totalRows) {
  std::vector<std::pair<size_t, size_t>> ranges;
  int overlap = 2;
  unsigned int rangeSize = (totalRows + overlap * (numRanks - 1)) / numRanks;
  unsigned int extraRows = (totalRows + overlap * (numRanks - 1)) % numRanks;

  size_t startRow = 0;
  for (int rank = 0; rank < numRanks; ++rank) {
    unsigned int currentRangeSize =
        rangeSize + (static_cast<unsigned int>(rank) >= static_cast<unsigned int>(numRanks) - extraRows ? 1 : 0);
    size_t endRow = startRow + currentRangeSize - 1;
    endRow = std::min(endRow, totalRows - 1);
    ranges.emplace_back(startRow, endRow);
    startRow = endRow - overlap + 1;
    if (startRow > totalRows - 1)
      startRow = totalRows - 1;
  }
  return ranges;
}

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
      << "  -e, --engine <name>          Engine: simple, simd, bitpack, cuda, cuda-colbatch, cuda-bitpack\n"
      << "  -h, --help                   Print this help message\n";
}

void printLine(size_t length) {
  for (size_t i = 0; i < length; i++)
    std::cout << "==";
  std::cout << "\n";
}

void printStep(const Grid &grid, const std::string &label, int value,
               float sleepTime) {
  std::cout << label << " " << value << "\n";
  printLine(grid.getNumCols());
  grid.printGrid();
  printLine(grid.getNumCols());
  if (sleepTime > 0) {
    std::cout.flush();
    std::cout << "\033[" << grid.getNumRows() + 3 << "A";
    usleep(static_cast<useconds_t>(sleepTime * 1.0e6));
  } else {
    std::cout.flush();
  }
}

bool initSimulation(int argc, char **argv, Grid &grid, SimParams &params) {
  const char *const short_opts = "f:r:s:g:n:p:o:e:h";
  const option long_opts[] = {{"file", required_argument, nullptr, 'f'},
                              {"random", required_argument, nullptr, 'r'},
                              {"seed", required_argument, nullptr, 's'},
                              {"generations", required_argument, nullptr, 'g'},
                              {"numthreads", required_argument, nullptr, 'n'},
                              {"print", optional_argument, nullptr, 'p'},
                              {"output", required_argument, nullptr, 'o'},
                              {"engine", required_argument, nullptr, 'e'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, no_argument, nullptr, 0}};

  std::string filePath;
  std::string randomArgs;
  bool seedProvided = false;

  int opt;
  while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) !=
         -1) {
    switch (opt) {
    case 'f':
      filePath = optarg;
      break;
    case 'r':
      randomArgs = optarg;
      break;
    case 's':
      params.seed = std::stoi(optarg);
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
      if      (eng == "simple")        params.engine = ENGINE_SIMPLE;
      else if (eng == "simd")         params.engine = ENGINE_SIMD;
      else if (eng == "bitpack")      params.engine = ENGINE_BITPACK;
      else if (eng == "cuda")         params.engine = ENGINE_CUDA;
      else if (eng == "cuda-colbatch") params.engine = ENGINE_CUDA_COLBATCH;
      else if (eng == "cuda-bitpack") params.engine = ENGINE_CUDA_BITPACK;
      else {
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

  if (!filePath.empty()) {
    grid = Grid(filePath);
    params.fullGridRows = grid.getNumRows();
    params.fullGridColumns = grid.getNumCols();
    params.alive = grid.aliveCells();
  } else if (!randomArgs.empty()) {
    auto values = parseArgs(randomArgs);
    if (values.size() != 3) {
      std::cerr << "Error: -r expects rows,cols,alive\n";
      return false;
    }
    params.fullGridRows = values[0];
    params.fullGridColumns = values[1];
    params.alive = values[2];
    params.randomInit = true;
    if (!seedProvided) params.seed = std::random_device()();

    // Bitpack engines construct BitGrid directly — skip 2.5GB Grid allocation
    bool needsGrid = (params.engine != ENGINE_BITPACK);
#if GOL_CUDA
    needsGrid = needsGrid && (params.engine != ENGINE_CUDA_BITPACK);
#endif
    if (needsGrid) {
      std::mt19937 rng(params.seed);
      grid = Grid(values[0], values[1], values[2], rng);
    }
  } else {
    std::cerr << "Error: No initialization option provided.\n";
    printHelp();
    return false;
  }

  std::cout << "Initializing GameOfLife with grid size (" << params.fullGridRows
            << ", " << params.fullGridColumns << ") and " << params.alive
            << " alive cells for " << params.steps << " generations\n";

  return true;
}

// ── MPI broadcast ───────────────────────────────────────────────────────────

void mpiBroadcastSimInfo(SimParams &params) {
  MPI_Bcast(&params.steps, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.numThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.fullGridRows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.fullGridColumns, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.sleepTime, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.engine, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

// ── Game-level MPI (use virtual base) ───────────────────────────────────────

void exchangeBoundaryRows(GameOfLife &game, int mpiRank, int mpiSize) {
  MPI_Datatype dtype = (game.getCellKind() == CellKind::Byte)
                           ? MPI_UINT8_T : MPI_UINT64_T;
  int stride = static_cast<int>(game.getStride());
  size_t rows = game.getNumRows();
  MPI_Request sendRequest[2];

  if (mpiRank > 0) {
    MPI_Isend(game.getRowDataRaw(1), stride, dtype, mpiRank - 1, 0,
              MPI_COMM_WORLD, &sendRequest[0]);
  }
  if (mpiRank < mpiSize - 1) {
    MPI_Isend(game.getRowDataRaw(rows - 2), stride, dtype,
              mpiRank + 1, 1, MPI_COMM_WORLD, &sendRequest[1]);
  }
  if (mpiRank > 0) {
    MPI_Recv(game.getRowDataRaw(0), stride, dtype, mpiRank - 1, 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (mpiRank < mpiSize - 1) {
    MPI_Recv(game.getRowDataRaw(rows - 1), stride, dtype,
             mpiRank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void assembleSend(GameOfLife &game, int mpiRank, int mpiSize) {
  MPI_Datatype dtype = (game.getCellKind() == CellKind::Byte)
                           ? MPI_UINT8_T : MPI_UINT64_T;
  int stride = static_cast<int>(game.getStride());
  size_t rows = game.getNumRows();
  size_t numRowsToSend = (mpiRank == mpiSize - 1) ? rows - 1 : rows - 2;
  for (size_t row = 1; row <= numRowsToSend; ++row) {
    MPI_Send(game.getRowDataRaw(row), stride, dtype, 0,
             static_cast<int>(row - 1), MPI_COMM_WORLD);
  }
}

template<typename GridType>
static Grid assembleFullGridImpl(GameOfLife &game, size_t fullRows,
                                 size_t fullCols, int mpiSize) {
  size_t localRows = game.getNumRows();
  int stride = static_cast<int>(game.getStride());
  const auto rowRanges = getRowRanges(mpiSize, fullRows);

  GridType assembled(fullRows, fullCols);
  for (size_t row = 0; row < localRows - 1; ++row)
    assembled.setRow(row,
        static_cast<const typename GridType::CellType *>(game.getRowDataRaw(row)));
  for (int rank = 1; rank < mpiSize; ++rank) {
    size_t startRow = rowRanges[rank].first + 1;
    size_t rowsToReceive =
        rowRanges[rank].second -
        ((rank == mpiSize - 1) ? startRow - 1 : startRow);
    for (size_t row = 0; row < rowsToReceive; ++row) {
      MPI_Recv(assembled.getRowData(startRow + row), stride,
               mpiDatatype<typename GridType::CellType>(),
               rank, static_cast<int>(row), MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
  if constexpr (std::is_same_v<GridType, Grid>)
    return assembled;
  else
    return assembled.toGrid();
}

Grid assembleFullGrid(GameOfLife &game, size_t fullRows, size_t fullCols,
                      int mpiSize) {
  if (game.getCellKind() == CellKind::Byte)
    return assembleFullGridImpl<Grid>(game, fullRows, fullCols, mpiSize);
  else
    return assembleFullGridImpl<BitGrid>(game, fullRows, fullCols, mpiSize);
}

// ── Grid-level MPI ──────────────────────────────────────────────────────────

template<typename GridType>
static void mpiSendGrid(const GridType &grid, int rank) {
  size_t rows = grid.getNumRows();
  size_t cols = grid.getNumCols();
  MPI_Send(&rows, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD);
  MPI_Send(&cols, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD);
  int totalElements = static_cast<int>(rows * grid.getStride());
  MPI_Send(grid.getData(), totalElements,
           mpiDatatype<typename GridType::CellType>(), rank, 0,
           MPI_COMM_WORLD);
}

template<typename GridType>
void mpiReceiveGrid(GridType &grid, int rank) {
  size_t rows, cols;
  MPI_Recv(&rows, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&cols, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  grid = GridType(rows, cols);
  int totalElements = static_cast<int>(rows * grid.getStride());
  MPI_Recv(grid.getData(), totalElements,
           mpiDatatype<typename GridType::CellType>(), rank, 0,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template void mpiReceiveGrid<Grid>(Grid &, int);
template void mpiReceiveGrid<BitGrid>(BitGrid &, int);

template<typename GridType>
void mpiSplitGrid(GridType &localGrid, const GridType &fullGrid, int mpiSize) {
  size_t gridRows = fullGrid.getNumRows();
  size_t gridColumns = fullGrid.getNumCols();
  const auto rowRanges = getRowRanges(mpiSize, gridRows);

  for (int rank = 0; rank < mpiSize; ++rank) {
    size_t startRow = rowRanges[rank].first;
    size_t endRow = rowRanges[rank].second;
    size_t localRows = endRow - startRow + 1;

    GridType chunk(localRows, gridColumns);
    for (size_t row = 0; row < localRows; ++row) {
      chunk.setRow(row, fullGrid.getRowData(startRow + row));
    }
    if (rank == 0) {
      localGrid = std::move(chunk);
    } else {
      mpiSendGrid(chunk, rank);
    }
  }
}

template void mpiSplitGrid<Grid>(Grid &, const Grid &, int);
template void mpiSplitGrid<BitGrid>(BitGrid &, const BitGrid &, int);
