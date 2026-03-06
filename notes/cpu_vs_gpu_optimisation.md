# CPU vs GPU Optimisation — Game of Life

## Why the CPU uses rows, but the GPU uses tiles

Both operate on the **same row-major flat array** in memory. The difference is how work is divided across threads.

### CPU: one row per thread

OpenMP distributes rows across cores with `#pragma omp parallel for`. Each thread walks its row sequentially, and the hardware prefetcher detects the linear access pattern and pre-fetches cache lines ahead of time. The 3×3 stencil reads from the previous, current, and next rows — all three fit in L1/L2 cache since they're adjacent in memory.

This works because:
- CPU cores are optimised for sequential access with deep cache hierarchies
- A small number of threads (4–128) each do a lot of work
- The prefetcher eliminates most memory latency

### GPU: 2D tiles in shared memory

A GPU launches thousands of threads grouped into blocks (e.g. 32×8 = 256 threads). Each block loads a tile of cells — plus a 1-cell halo — from global memory (DRAM) into **shared memory** (on-chip SRAM, ~100× faster than global memory). After a `__syncthreads()` barrier, every thread computes its cell's next state entirely from shared memory.

This is necessary because:

1. **Redundant global reads.** Each cell is a neighbour of up to 8 others. Without shared memory, every thread independently fetches its 8 neighbours from DRAM. With a tile, each cell is loaded from global memory **once**, then read from shared memory up to 9 times.

2. **Coalesced memory access.** GPU memory controllers combine 32 consecutive threads reading 32 consecutive addresses into a single memory transaction. With `threadIdx.x` = column and `blockDim.x = 32`, each warp reads one contiguous row segment — perfectly coalesced. A row-per-thread layout would have 32 threads reading from 32 different rows — completely uncoalesced, wasting bandwidth.

3. **2D locality matches the 2D stencil.** The 3×3 neighbourhood spans both rows and columns. A 2D tile keeps the entire neighbourhood spatially local in shared memory. The halo overhead is small: a 32×8 tile needs (32+2)×(8+2) = 340 bytes, trivial for the 48KB+ shared memory per streaming multiprocessor.

### Key insight

The **global memory layout doesn't change** — it's still flat row-major (`src[row * cols + col]`). Tiles exist only transiently in shared memory during kernel execution. This avoids the downsides of a tiled *storage* format on CPU (broken prefetcher, cross-tile cache misses) while gaining the GPU benefits of reduced global memory traffic and coalesced access.

| Aspect | CPU | GPU |
|---|---|---|
| Parallelism | Few threads, lots of work each | Thousands of threads, little work each |
| Memory hierarchy | L1 → L2 → L3 → DRAM | Registers → Shared → L2 → DRAM |
| Preferred access | Sequential (prefetcher) | Coalesced (32-wide transactions) |
| Work unit | Row | 2D tile |
| Data reuse strategy | Cache lines (implicit) | Shared memory (explicit) |
| Storage layout | Row-major | Row-major (unchanged) |
