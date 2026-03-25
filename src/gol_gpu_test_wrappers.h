#ifndef GOL_GPU_TEST_WRAPPERS_H
#define GOL_GPU_TEST_WRAPPERS_H

#include <cstddef>
#include <cstdint>

// Free functions that run a single kernel step on host data, bypassing the
// GPUEngine class.  Each function allocates device buffers, copies input
// host-to-device, launches the kernel with the same configuration as the
// corresponding engine, copies the result device-to-host, and frees the
// device buffers.
//
// These wrappers allow isolated testing of the GPU kernels without the
// GPUEngine ownership / double-buffer lifecycle.

#if GOL_CUDA

void cudaSimpleKernelStep(const uint8_t *in, uint8_t *out, size_t rows,
                          size_t cols);

void cudaTileKernelStep(const uint8_t *in, uint8_t *out, size_t rows,
                        size_t cols);

void cudaBitPackKernelStep(const uint64_t *in, uint64_t *out, size_t rows,
                           size_t stride, size_t cols);

#endif // GOL_CUDA

#if GOL_HIP

void hipSimpleKernelStep(const uint8_t *in, uint8_t *out, size_t rows,
                         size_t cols);

void hipTileKernelStep(const uint8_t *in, uint8_t *out, size_t rows,
                       size_t cols);

void hipBitPackKernelStep(const uint64_t *in, uint64_t *out, size_t rows,
                          size_t stride, size_t cols);

#endif // GOL_HIP

#endif // GOL_GPU_TEST_WRAPPERS_H
