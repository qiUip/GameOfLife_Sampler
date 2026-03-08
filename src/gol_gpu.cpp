#include "gol_gpu.h"

template <typename CellT, typename HostGridT>
GPUEngine<CellT, HostGridT>::GPUEngine(HostGridT hostGrid, const GpuOps &ops)
    : ops_(ops), hostGrid_(std::move(hostGrid)),
      stride_(hostGrid_.getStride()) {
    rows_             = hostGrid_.getNumRows();
    cols_             = hostGrid_.getNumCols();
    size_t totalBytes = rows_ * stride_ * sizeof(CellT);
    ops_.alloc(reinterpret_cast<void **>(&d_current_), totalBytes);
    ops_.alloc(reinterpret_cast<void **>(&d_next_), totalBytes);
    ops_.copyH2D(d_current_, hostGrid_.getData(), totalBytes);
}

template <typename CellT, typename HostGridT>
GPUEngine<CellT, HostGridT>::~GPUEngine() {
    ops_.free(d_current_);
    ops_.free(d_next_);
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::takeStep() {
    launchKernel(d_current_, d_next_);
    ops_.checkLast("takeStep");
    CellT *tmp = d_current_;
    d_current_ = d_next_;
    d_next_    = tmp;
}

template <typename CellT, typename HostGridT>
void *GPUEngine<CellT, HostGridT>::getRowDataRaw(size_t row) {
    ops_.copyD2H(hostGrid_.getRowData(row), d_current_ + row * stride_,
                 stride_ * sizeof(CellT));
    return hostGrid_.getRowData(row);
}

template <typename CellT, typename HostGridT>
size_t GPUEngine<CellT, HostGridT>::getStride() const {
    return stride_;
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::commitBoundaries() {
    ops_.copyH2D(d_current_, hostGrid_.getRowData(0), stride_ * sizeof(CellT));
    ops_.copyH2D(d_current_ + (rows_ - 1) * stride_,
                 hostGrid_.getRowData(rows_ - 1), stride_ * sizeof(CellT));
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::sync() {
    ops_.sync();
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::printGrid() const {
    ops_.copyD2H(hostGrid_.getData(), d_current_,
                 rows_ * stride_ * sizeof(CellT));
    hostGrid_.printGrid();
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::writeToFile(
    const std::string &filename) const {
    ops_.copyD2H(hostGrid_.getData(), d_current_,
                 rows_ * stride_ * sizeof(CellT));
    hostGrid_.writeToFile(filename);
}

// Explicit instantiations
template class GPUEngine<uint8_t, Grid>;
template class GPUEngine<uint64_t, BitGrid>;
