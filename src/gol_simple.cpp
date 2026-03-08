#include "gol_engine.h"

#include <cstring>

SimpleGameOfLife::SimpleGameOfLife(Grid &grid)
    : currentGrid_(std::move(grid)),
      newGrid_(currentGrid_.getNumRows(), currentGrid_.getNumCols()) {
    rows_ = currentGrid_.getNumRows();
    cols_ = currentGrid_.getNumCols();
}

void SimpleGameOfLife::takeStep() {
    const uint8_t *src  = currentGrid_.getData();
    uint8_t *dst        = newGrid_.getData();
    const size_t rows   = rows_;
    const size_t cols   = cols_;
    const size_t maxRow = rows - 1;
    const size_t maxCol = cols - 1;

    constexpr int8_t offsets[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                      {0, 1},   {1, -1}, {1, 0},  {1, 1}};

#pragma omp parallel for
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            size_t count = 0;
            for (const auto &offset : offsets) {
                const size_t nr = row + offset[0];
                const size_t nc = col + offset[1];
                if (nr <= maxRow && nc <= maxCol) {
                    count += src[nr * cols + nc];
                }
            }

            bool alive = src[row * cols + col];
            if (!alive && count == 3) {
                alive = true;
            } else if (alive && (count < 2 || count > 3)) {
                alive = false;
            }
            dst[row * cols + col] = alive;
        }
    }

    currentGrid_.swap(newGrid_);
}

void *SimpleGameOfLife::getRowDataRaw(size_t row) {
    return currentGrid_.getRowData(row);
}

size_t SimpleGameOfLife::getStride() const {
    return cols_;
}

CellKind SimpleGameOfLife::getCellKind() const {
    return CellKind::Byte;
}
