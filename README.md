# Game of Life Sampler

A Conway's Game of Life implementation exploring CPU and GPU optimisation
strategies, with MPI domain decomposition for multi-node execution.

## Engines

| Engine         | Description                                                  |
|----------------|--------------------------------------------------------------|
| `simple`       | Scalar loop with OpenMP threading                            |
| `simd`         | Compile-time SIMD dispatch (std::simd, NEON, AVX2, AVX512)  |
| `bitpack`      | Bit-packed grid with 64-bit parallel neighbour counting      |
| `cuda-simple`  | CUDA kernel with interior/border split                       |
| `cuda-tile`    | CUDA tiled kernel with shared memory and vectorised loads    |
| `cuda-bitpack` | CUDA bit-packed kernel with shared memory                    |
| `hip-simple`   | HIP kernel with interior/border split                        |
| `hip-tile`     | HIP tiled kernel with shared memory and vectorised loads     |
| `hip-bitpack`  | HIP bit-packed kernel with shared memory                     |

## Build

### CPU only (default)

```
cmake -B build
cmake --build build
```

### CUDA

If CUDA is present on the system, it should be auto-detected via
`find_package(CUDAToolkit)`. GPU architecture detection (`native`) requires
CMake 3.24+. Specific GPU architecture can also be targetted manually with,
e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` for Ampere.

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build
```

### HIP

If HIP is present on the system, it should be auto-detected via
`find_package(hip)`. GPU architecture detection (`native`) requires CMake 3.24+
and `rocm_agent_enumerator` on `PATH`. Specific GPU architecture can also be
targetted manually with, e.g. `-DCMAKE_HIP_ARCHITECTURES=gfx942` for
MI300A/MI300X.

```
cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build
```

## Run

```
./build/golSimulator -r 1000,1000,50000 -g 100 -p -1 -e simd
```

This creates a 1000x1000 grid with 50,000 alive cells, runs 100 generations
using the SIMD engine without printing the grid to screen.  

Use `-h` for the full list of options.

## Tests

```
ctest --test-dir build
```
