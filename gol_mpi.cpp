#include "gol_mpi.h"

#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <type_traits>

// ── MPI datatype trait ──────────────────────────────────────────────────────

template <typename T> static MPI_Datatype mpiDatatype();
template <> MPI_Datatype mpiDatatype<uint8_t>() {
    return MPI_UINT8_T;
}
template <> MPI_Datatype mpiDatatype<uint64_t>() {
    return MPI_UINT64_T;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::vector<std::pair<size_t, size_t>> getRowRanges(int numRanks,
                                                           size_t totalRows) {
    std::vector<std::pair<size_t, size_t>> ranges;
    int overlap            = 2;
    unsigned int rangeSize = (totalRows + overlap * (numRanks - 1)) / numRanks;
    unsigned int extraRows = (totalRows + overlap * (numRanks - 1)) % numRanks;

    size_t startRow = 0;
    for (int rank = 0; rank < numRanks; ++rank) {
        unsigned int currentRangeSize =
            rangeSize + (static_cast<unsigned int>(rank) >=
                                 static_cast<unsigned int>(numRanks) - extraRows
                             ? 1
                             : 0);
        size_t endRow = startRow + currentRangeSize - 1;
        endRow        = std::min(endRow, totalRows - 1);
        ranges.emplace_back(startRow, endRow);
        startRow = endRow - overlap + 1;
        if (startRow > totalRows - 1)
            startRow = totalRows - 1;
    }
    return ranges;
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
    MPI_Datatype dtype =
        (game.getCellKind() == CellKind::Byte) ? MPI_UINT8_T : MPI_UINT64_T;
    int stride  = static_cast<int>(game.getStride());
    size_t rows = game.getNumRows();
    MPI_Request sendRequest[2];

    if (mpiRank > 0) {
        MPI_Isend(game.getRowDataRaw(1), stride, dtype, mpiRank - 1, 0,
                  MPI_COMM_WORLD, &sendRequest[0]);
    }
    if (mpiRank < mpiSize - 1) {
        MPI_Isend(game.getRowDataRaw(rows - 2), stride, dtype, mpiRank + 1, 1,
                  MPI_COMM_WORLD, &sendRequest[1]);
    }
    if (mpiRank > 0) {
        MPI_Recv(game.getRowDataRaw(0), stride, dtype, mpiRank - 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (mpiRank < mpiSize - 1) {
        MPI_Recv(game.getRowDataRaw(rows - 1), stride, dtype, mpiRank + 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void assembleSend(GameOfLife &game, int mpiRank, int mpiSize) {
    MPI_Datatype dtype =
        (game.getCellKind() == CellKind::Byte) ? MPI_UINT8_T : MPI_UINT64_T;
    int stride           = static_cast<int>(game.getStride());
    size_t rows          = game.getNumRows();
    size_t numRowsToSend = (mpiRank == mpiSize - 1) ? rows - 1 : rows - 2;
    for (size_t row = 1; row <= numRowsToSend; ++row) {
        MPI_Send(game.getRowDataRaw(row), stride, dtype, 0,
                 static_cast<int>(row - 1), MPI_COMM_WORLD);
    }
}

template <typename GridType>
static GridType assembleFullGridImpl(GameOfLife &game, size_t fullRows,
                                     size_t fullCols, int mpiSize) {
    size_t localRows     = game.getNumRows();
    int stride           = static_cast<int>(game.getStride());
    const auto rowRanges = getRowRanges(mpiSize, fullRows);

    GridType assembled(fullRows, fullCols);
    for (size_t row = 0; row < localRows - 1; ++row)
        assembled.setRow(row, static_cast<const typename GridType::CellType *>(
                                  game.getRowDataRaw(row)));
    for (int rank = 1; rank < mpiSize; ++rank) {
        size_t startRow = rowRanges[rank].first + 1;
        size_t rowsToReceive =
            rowRanges[rank].second -
            ((rank == mpiSize - 1) ? startRow - 1 : startRow);
        for (size_t row = 0; row < rowsToReceive; ++row) {
            MPI_Recv(assembled.getRowData(startRow + row), stride,
                     mpiDatatype<typename GridType::CellType>(), rank,
                     static_cast<int>(row), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    return assembled;
}

template <typename GridType>
static void assembleOutputImpl(GameOfLife &game, size_t fullRows,
                               size_t fullCols, int mpiSize, float sleepTime,
                               int step, const std::string &outfile) {
    GridType grid =
        assembleFullGridImpl<GridType>(game, fullRows, fullCols, mpiSize);
    if (sleepTime >= 0)
        printStep(grid, "Generation:", step, sleepTime);
    if (!outfile.empty())
        grid.writeToFile(outfile);
}

void assembleOutput(GameOfLife &game, size_t fullRows, size_t fullCols,
                    int mpiSize, float sleepTime, int step,
                    const std::string &outfile) {
    if (game.getCellKind() == CellKind::Byte)
        assembleOutputImpl<Grid>(game, fullRows, fullCols, mpiSize, sleepTime,
                                 step, outfile);
    else
        assembleOutputImpl<BitGrid>(game, fullRows, fullCols, mpiSize,
                                    sleepTime, step, outfile);
}

// ── Grid-level MPI ──────────────────────────────────────────────────────────

template <typename GridType>
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

template <typename GridType> void mpiReceiveGrid(GridType &grid, int rank) {
    size_t rows, cols;
    MPI_Recv(&rows, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(&cols, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    grid              = GridType(rows, cols);
    int totalElements = static_cast<int>(rows * grid.getStride());
    MPI_Recv(grid.getData(), totalElements,
             mpiDatatype<typename GridType::CellType>(), rank, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template void mpiReceiveGrid<Grid>(Grid &, int);
template void mpiReceiveGrid<BitGrid>(BitGrid &, int);

template <typename GridType>
void mpiSplitGrid(GridType &localGrid, const GridType &fullGrid, int mpiSize) {
    size_t gridRows      = fullGrid.getNumRows();
    size_t gridColumns   = fullGrid.getNumCols();
    const auto rowRanges = getRowRanges(mpiSize, gridRows);

    for (int rank = 0; rank < mpiSize; ++rank) {
        size_t startRow  = rowRanges[rank].first;
        size_t endRow    = rowRanges[rank].second;
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
