#include "gol_gpu.h"

// Device stride is padded to ensure uint32_t alignment for vectorised
// loads/stores in tiled byte kernels.  For byte grids (CellT = uint8_t),
// the stride is rounded up to a multiple of 4.  For word grids
// (CellT = uint64_t), each element is already 8-byte aligned.
template <typename CellT> static size_t paddedStride(size_t hostStride) {
    if constexpr (sizeof(CellT) == 1)
        return (hostStride + 3) & ~size_t(3);
    else
        return hostStride;
}

template <typename CellT, typename HostGridT>
GPUEngine<CellT, HostGridT>::GPUEngine(HostGridT hostGrid, const GpuOps &ops)
    : ops_(ops), hostGrid_(std::move(hostGrid)), stride_(hostGrid_.getStride()),
      deviceStride_(paddedStride<CellT>(stride_)) {
    rows_             = hostGrid_.getNumRows();
    cols_             = hostGrid_.getNumCols();
    size_t totalBytes = rows_ * deviceStride_ * sizeof(CellT);
    ops_.alloc(reinterpret_cast<void **>(&d_current_), totalBytes);
    ops_.alloc(reinterpret_cast<void **>(&d_next_), totalBytes);

    // Zero both buffers so padding cells are dead.
    ops_.memset(d_current_, 0, totalBytes);
    ops_.memset(d_next_, 0, totalBytes);

    // Pitched copy — host and device strides may differ.
    size_t rowBytes = stride_ * sizeof(CellT);
    size_t devPitch = deviceStride_ * sizeof(CellT);
    ops_.copy2D_H2D(d_current_, devPitch, hostGrid_.getData(), rowBytes,
                    rowBytes, rows_);
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
    ops_.copyD2H(hostGrid_.getRowData(row), d_current_ + row * deviceStride_,
                 stride_ * sizeof(CellT));
    return hostGrid_.getRowData(row);
}

template <typename CellT, typename HostGridT>
size_t GPUEngine<CellT, HostGridT>::getStride() const {
    return stride_;
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::commitBoundaries() {
    size_t rowBytes = stride_ * sizeof(CellT);
    ops_.copyH2D(d_current_, hostGrid_.getRowData(0), rowBytes);
    ops_.copyH2D(d_current_ + (rows_ - 1) * deviceStride_,
                 hostGrid_.getRowData(rows_ - 1), rowBytes);
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::sync() {
    ops_.sync();
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::printGrid() const {
    size_t rowBytes = stride_ * sizeof(CellT);
    size_t devPitch = deviceStride_ * sizeof(CellT);
    ops_.copy2D_D2H(hostGrid_.getData(), rowBytes, d_current_, devPitch,
                    rowBytes, rows_);
    hostGrid_.printGrid();
}

template <typename CellT, typename HostGridT>
void GPUEngine<CellT, HostGridT>::writeToFile(
    const std::string &filename) const {
    size_t rowBytes = stride_ * sizeof(CellT);
    size_t devPitch = deviceStride_ * sizeof(CellT);
    ops_.copy2D_D2H(hostGrid_.getData(), rowBytes, d_current_, devPitch,
                    rowBytes, rows_);
    hostGrid_.writeToFile(filename);
}

// Explicit instantiations
template class GPUEngine<uint8_t, Grid>;
template class GPUEngine<uint64_t, BitGrid>;
