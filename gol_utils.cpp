#include "gol_utils.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>

#include <getopt.h>
#include <mpi.h>
#include <unistd.h>

// Helper: parse comma-separated values
static std::vector<size_t> parseArgs(const std::string &args) {
  std::vector<size_t> values;
  std::istringstream ss(args);
  std::string token;
  while (std::getline(ss, token, ',')) {
    values.push_back(std::stoul(token));
  }
  return values;
}

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
  printLine(grid.getNumColumns());
  grid.printGrid();
  printLine(grid.getNumColumns());
  if (sleepTime > 0) {
    std::cout.flush();
    std::cout << "\033[" << grid.getNumRows() + 3 << "A";
    usleep(static_cast<useconds_t>(sleepTime * 1.0e6));
  } else {
    std::cout.flush();
  }
}

bool initSimulation(int argc, char **argv, Grid &grid, SimParams &params) {
  const char *const short_opts = "f:r:s:g:n:p:o:h";
  const option long_opts[] = {{"file", required_argument, nullptr, 'f'},
                              {"random", required_argument, nullptr, 'r'},
                              {"seed", required_argument, nullptr, 's'},
                              {"generations", required_argument, nullptr, 'g'},
                              {"numthreads", required_argument, nullptr, 'n'},
                              {"print", optional_argument, nullptr, 'p'},
                              {"output", required_argument, nullptr, 'o'},
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
    case 'h':
      printHelp();
      return false;
    default:
      printHelp();
      return false;
    }
  }

  // Initialize grid
  if (!filePath.empty()) {
    grid = Grid(filePath);
  } else if (!randomArgs.empty()) {
    auto values = parseArgs(randomArgs);
    if (values.size() != 3) {
      std::cerr << "Error: -r expects rows,cols,alive\n";
      return false;
    }
    std::mt19937 rng(seedProvided ? params.seed : std::random_device()());
    grid = Grid(values[0], values[1], values[2], rng);
  } else {
    std::cerr << "Error: No initialization option provided.\n";
    printHelp();
    return false;
  }

  params.fullGridRows = grid.getNumRows();
  params.fullGridColumns = grid.getNumColumns();

  std::cout << "Initializing GameOfLife with grid size (" << params.fullGridRows
            << ", " << params.fullGridColumns << ") and " << grid.aliveCells()
            << " alive cells for " << params.steps << " generations\n";

  return true;
}

void mpiBroadcastSimInfo(SimParams &params) {
  MPI_Bcast(&params.steps, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.numThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.fullGridRows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.fullGridColumns, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params.sleepTime, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// Helper: calculate row ranges for domain decomposition
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

void mpiSendNewGrid(const Grid &grid, int rank) {
  size_t gridRows = grid.getNumRows();
  size_t gridColumns = grid.getNumColumns();
  MPI_Send(&gridRows, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD);
  MPI_Send(&gridColumns, 1, MPI_UNSIGNED_LONG, rank, 0, MPI_COMM_WORLD);
  size_t totalCells = gridRows * gridColumns;
  MPI_Send(grid.getCellsPointer(), totalCells, MPI_UINT8_T, rank, 0,
           MPI_COMM_WORLD);
}

void mpiReceiveNewGrid(Grid &grid, int rank) {
  size_t gridRows, gridColumns;
  MPI_Recv(&gridRows, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&gridColumns, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  size_t totalCells = gridRows * gridColumns;
  grid = Grid(gridRows, gridColumns);
  MPI_Recv(grid.getCellsPointer(), totalCells, MPI_UINT8_T, rank, 0,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void mpiSplitGrid(Grid &fullGrid, int mpiSize) {
  size_t gridRows = fullGrid.getNumRows();
  size_t gridColumns = fullGrid.getNumColumns();
  const auto rowRanges = getRowRanges(mpiSize, gridRows);

  Grid grid;
  for (int rank = 0; rank < mpiSize; ++rank) {
    size_t startRow = rowRanges[rank].first;
    size_t endRow = rowRanges[rank].second;

    Grid localGrid(endRow - startRow + 1, gridColumns);
    for (size_t row = startRow; row <= endRow; ++row) {
      localGrid.setRow(row - startRow, fullGrid.getRowPointer(row));
    }
    if (rank == 0) {
      grid = std::move(localGrid);
    } else {
      mpiSendNewGrid(localGrid, rank);
    }
  }
  fullGrid = std::move(grid);
}

void exchangeBoundaryRows(GameOfLife &game, size_t gridRows, size_t gridColumns,
                          int mpiRank, int mpiSize) {
  MPI_Request sendRequest[2];
  MPI_Status recvStatus[2];

  if (mpiRank > 0) {
    MPI_Isend(game.getRowPointer(1), gridColumns, MPI_UINT8_T, mpiRank - 1, 0,
              MPI_COMM_WORLD, &sendRequest[0]);
  }
  if (mpiRank < mpiSize - 1) {
    MPI_Isend(game.getRowPointer(gridRows - 2), gridColumns, MPI_UINT8_T,
              mpiRank + 1, 1, MPI_COMM_WORLD, &sendRequest[1]);
  }
  if (mpiRank > 0) {
    MPI_Recv(game.getRowPointer(0), gridColumns, MPI_UINT8_T, mpiRank - 1, 1,
             MPI_COMM_WORLD, &recvStatus[0]);
  }
  if (mpiRank < mpiSize - 1) {
    MPI_Recv(game.getRowPointer(gridRows - 1), gridColumns, MPI_UINT8_T,
             mpiRank + 1, 0, MPI_COMM_WORLD, &recvStatus[1]);
  }
}

void assembleGridSend(const Grid &grid, size_t gridRows, size_t gridColumns,
                      int mpiRank, int mpiSize) {
  size_t numRowsToSend = (mpiRank == mpiSize - 1) ? gridRows - 1 : gridRows - 2;
  for (size_t row = 1; row <= numRowsToSend; ++row) {
    MPI_Send(grid.getRowPointer(row), gridColumns, MPI_UINT8_T, 0, row - 1,
             MPI_COMM_WORLD);
  }
}

Grid assembleGrid(const Grid &grid, size_t gridRows, size_t gridColumns,
                  size_t fullGridRows, size_t fullGridColumns, int mpiRank,
                  int mpiSize) {
  Grid assembledGrid(fullGridRows, fullGridColumns);
  for (size_t row = 0; row < gridRows - 1; ++row) {
    assembledGrid.setRow(row, grid.getRowPointer(row));
  }
  const auto rowRanges = getRowRanges(mpiSize, fullGridRows);
  for (int rank = 1; rank < mpiSize; ++rank) {
    size_t startRow = rowRanges[rank].first + 1;
    size_t rowsToReceive =
        rowRanges[rank].second -
        ((rank == mpiSize - 1) ? startRow - 1 : startRow);
    for (size_t row = 0; row < rowsToReceive; ++row) {
      MPI_Recv(assembledGrid.getRowPointer(startRow + row), gridColumns,
               MPI_UINT8_T, rank, row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  return assembledGrid;
}
