#ifndef GOL_GPU_H
#define GOL_GPU_H

#include "gol.h"

// GPU API abstraction — function pointers avoid virtual dispatch pitfalls
// in constructors and have negligible overhead since GPU API calls are
// never in the hot path.
struct GpuOps {
  void (*alloc)(void **ptr, size_t bytes);
  void (*free)(void *ptr);
  void (*copyH2D)(void *dst, const void *src, size_t bytes);
  void (*copyD2H)(void *dst, const void *src, size_t bytes);
  void (*sync)();
  void (*checkLast)(const char *context);
};

// Shared GPU engine base — owns device buffers, handles memcpy boilerplate.
// Derived classes only override launchKernel() and getCellKind().
template<typename CellT, typename HostGridT>
class GPUEngine : public GameOfLife {
public:
  ~GPUEngine();
  void takeStep() override;
  void  *getRowDataRaw(size_t row) override;
  size_t getStride() const override;
  void commitBoundaries() override;
  void sync() override;
  void printGrid() const override;
  void writeToFile(const std::string &filename) const override;

protected:
  explicit GPUEngine(HostGridT hostGrid, const GpuOps &ops);
  virtual void launchKernel(const CellT *src, CellT *dst) = 0;

  GpuOps ops_;
  CellT *d_current_, *d_next_;
  mutable HostGridT hostGrid_;
  size_t stride_;
};

extern template class GPUEngine<uint8_t, Grid>;
extern template class GPUEngine<uint64_t, BitGrid>;

// ── CUDA engines ────────────────────────────────────────────────────────────

#if GOL_CUDA

class CUDASimpleGameOfLife : public GPUEngine<uint8_t, Grid> {
public:
  explicit CUDASimpleGameOfLife(Grid &grid);
  CellKind getCellKind() const override;
protected:
  void launchKernel(const uint8_t *src, uint8_t *dst) override;
};

class CUDATileGameOfLife : public GPUEngine<uint8_t, Grid> {
public:
  explicit CUDATileGameOfLife(Grid &grid);
  CellKind getCellKind() const override;
protected:
  void launchKernel(const uint8_t *src, uint8_t *dst) override;
};

class CUDABitPackGameOfLife : public GPUEngine<uint64_t, BitGrid> {
public:
  explicit CUDABitPackGameOfLife(Grid &grid);
  explicit CUDABitPackGameOfLife(BitGrid &grid);
  CellKind getCellKind() const override;
protected:
  void launchKernel(const uint64_t *src, uint64_t *dst) override;
};

#endif // GOL_CUDA

// ── HIP engines ─────────────────────────────────────────────────────────────

#if GOL_HIP

class HIPSimpleGameOfLife : public GPUEngine<uint8_t, Grid> {
public:
  explicit HIPSimpleGameOfLife(Grid &grid);
  CellKind getCellKind() const override;
protected:
  void launchKernel(const uint8_t *src, uint8_t *dst) override;
};

class HIPTileGameOfLife : public GPUEngine<uint8_t, Grid> {
public:
  explicit HIPTileGameOfLife(Grid &grid);
  CellKind getCellKind() const override;
protected:
  void launchKernel(const uint8_t *src, uint8_t *dst) override;
};

class HIPBitPackGameOfLife : public GPUEngine<uint64_t, BitGrid> {
public:
  explicit HIPBitPackGameOfLife(Grid &grid);
  explicit HIPBitPackGameOfLife(BitGrid &grid);
  CellKind getCellKind() const override;
protected:
  void launchKernel(const uint64_t *src, uint64_t *dst) override;
};

#endif // GOL_HIP

#endif // GOL_GPU_H
