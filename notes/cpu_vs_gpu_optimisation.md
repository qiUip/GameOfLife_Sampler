# Optimising Conway's Game of Life: From Scalar CPU to GPU

This document describes five implementations of Conway's Game of Life, each applying
a different set of optimisation techniques. All implementations compute the same result:
for every cell in a 2D grid, count its eight neighbours and apply the birth/survival rule.
The implementations differ in data layout, instruction-level strategy, parallelism model,
and memory hierarchy usage.

Benchmark: 50,000 x 50,000 grid, 100 generations.

| # | Engine | Hardware | Time | Speedup |
|---|--------|----------|------|---------|
| 1 | Simple | 30-core CPU | 83.0 s | 1x |
| 2 | SIMD | 30-core CPU | 14.5 s | 5.7x |
| 3 | Bit-packed | 30-core CPU | 1.86 s | 44.6x |
| 4 | CUDA byte | GPU (V100) | 1.41 s | 58.9x |
| 5 | CUDA bit-packed | GPU (V100) | 0.12 s | 664x |

---

## 1. Simple (baseline CPU)

**File:** `gol_simple.cpp`

### Cell representation

Each cell is one `uint8_t` byte, storing either 0 (dead) or 1 (alive). The grid is a
flat row-major array: cell (r, c) is at `grid[r * cols + c]`.

This is deliberately wasteful — a cell only needs one bit, not eight — but it means
each cell has its own address. Any cell can be read or written with a single byte load
or store, with no masking or shifting.

### Neighbour counting

The eight neighbours of cell (r, c) are visited via a hardcoded offset table:

```
offsets = {(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)}
```

For each offset, the code computes the neighbour's row and column, checks whether it
falls within the grid, and if so, adds the neighbour's value to a running count. The
boundary check uses unsigned arithmetic: when row 0 adds offset -1, the `size_t` result
wraps to a large positive number that fails the `<= maxRow` test.

### Branching

The Game of Life rule is expressed with explicit if/else branches:

```cpp
if (!alive && count == 3)
    alive = true;
else if (alive && (count < 2 || count > 3))
    alive = false;
```

This is clear and readable. It is also slow: the branch predictor cannot predict the
outcome because it depends on the cell's neighbours, which vary essentially at random.
Every misprediction flushes the CPU pipeline (typically 15-20 cycles on a modern
out-of-order core). With 8 branches per cell (one per neighbour boundary check) plus
the rule branches, mispredictions accumulate.

### Parallelism

OpenMP parallelises the outer loop over rows: `#pragma omp parallel for`. Each thread
processes a contiguous band of rows. Since the output grid is separate from the input
grid (double-buffered), threads do not interfere.

### Memory access

Each thread walks its rows left to right. The CPU hardware prefetcher detects this
sequential pattern and fetches cache lines ahead of the read position. Three rows are
active simultaneously (previous, current, next), and since they are adjacent in memory,
they typically share L1 or L2 cache.

### Performance costs

The primary costs are:
1. **Branch mispredictions** on every boundary check and on the GoL rule.
2. **Byte-at-a-time processing** — the CPU's SIMD units sit idle.
3. **One cell per iteration** — no data-level parallelism is exploited.

---

## 2. SIMD (vectorised CPU)

**File:** `gol_simd.cpp`

### Cell representation

Identical to the simple engine: one `uint8_t` per cell, flat row-major layout.

### Key idea: process many cells per instruction

Instead of computing one cell at a time, the SIMD engine loads a vector of consecutive
cells (16 on ARM NEON, 32 on AVX2, 64 on AVX-512) and performs the neighbour sum and
GoL rule on all of them simultaneously, using a single instruction per operation.

### Neighbour counting — the three-pointer technique

For an interior row, the engine maintains three pointers: `p` (previous row), `c`
(current row), `n` (next row). To count neighbours for cells at columns [col, col+W),
it performs nine vector loads:

```
pL = load(p + col - 1)    pC = load(p + col)    pR = load(p + col + 1)
cL = load(c + col - 1)    cC = load(c + col)    cR = load(c + col + 1)
nL = load(n + col - 1)    nC = load(n + col)    nR = load(n + col + 1)
```

The neighbour count vector is `nb = pL + pC + pR + cL + cR + nL + nC + nR`, computed
with eight vector additions. Each lane of `nb` holds the neighbour count (0-8) for one
cell.

The loads at `col - 1` and `col + 1` are offset by one byte from `col`. This means
each cell's left and right neighbours are naturally picked up by the shifted loads.
No shuffling or permuting is needed — the byte-per-cell layout makes the shift free.

### Branch-free GoL rule

The rule is expressed as a vector comparison, not a branch:

```
born    = (nb == 3)
survive = (cC != 0) & (nb == 2)
result  = (born | survive) & 1
```

The comparison instructions produce all-ones (0xFF) or all-zeros (0x00) masks. The
bitwise AND with 1 converts these to the 0/1 cell values. No branches are executed.
The CPU processes all W cells in the vector with the same instruction sequence
regardless of their values.

### Boundary handling with templates

The first and last rows lack a previous or next row. The first and last columns lack
left or right neighbours. Rather than adding runtime boundary checks to the inner loop
(which would prevent vectorisation), the engine uses C++ templates:

```cpp
template<bool HasPrev, bool HasNext, bool HasLeft, bool HasRight>
uint8_t aliveNeighbours(...)
```

The compiler generates specialised versions for each boundary combination. At compile
time, neighbour accesses that would be out of bounds are omitted entirely. The top row
calls `processBorderRow<false, true>`, the bottom row calls `processBorderRow<true,
false>`, and all interior rows call `processInteriorRow` which invokes the fully
vectorised `processInteriorCells` for the interior columns.

Since a 50,000-row grid has only 2 border rows and 49,998 interior rows, the scalar
boundary code handles 0.004% of the work.

### Why it is faster

The speedup over the simple engine comes from three sources:
1. **Data-level parallelism.** Each vector instruction processes 16-64 cells.
2. **No branch mispredictions.** The rule and boundary checks are branch-free.
3. **Efficient memory access.** Sequential loads feed the prefetcher; unaligned SIMD
   loads (`loadu`) handle the +-1 offsets with no penalty on modern CPUs.

---

## 3. Bit-packed (CPU)

**File:** `gol_bitpack.cpp`

### Cell representation

Each cell is one bit. Sixty-four cells are packed into a single `uint64_t` word. The
grid is stored as an array of words: row `r`, columns [64w, 64w+63] are in
`grid[r * wordsPerRow + w]`. Bit 0 is the leftmost cell in the word.

A 50,000 x 50,000 grid occupies 50,000 x 782 x 8 = 313 MB, compared to 2.5 GB for
the byte-per-cell layout — an 8x reduction in memory footprint.

### Why bit-packing changes the algorithm

With byte-per-cell storage, neighbour counting is addition: load 8 bytes, add them.
With bit-per-cell storage, each byte holds 8 cells, so a normal add would mix different
cells' values. The algorithm must instead count neighbours using **bitwise logic** that
operates on all 64 cells in a word simultaneously.

### Neighbour counting — binary addition circuits

The engine builds a per-bit population count using the same logic as a hardware adder
circuit. The key primitive is `rowSum3`, which takes three 64-bit words (left-shifted,
centre, right-shifted) and produces a 2-bit sum per bit position:

```cpp
void rowSum3(uint64_t L, uint64_t C, uint64_t R,
             uint64_t &s1, uint64_t &s0) {
    uint64_t t = L ^ C;
    s0 = t ^ R;           // bit 0 of sum
    s1 = (L & C) | (t & R); // bit 1 of sum
}
```

This is a full adder: for each of the 64 bit positions independently, it computes the
sum of 3 input bits as a 2-bit result (s1:s0). The XOR computes the sum bit; the
AND/OR combination computes the carry bit. This processes 64 cells with 5 bitwise
instructions.

The three row sums (previous, current, next) are combined by `sum9`, which adds three
2-bit numbers into a 4-bit result (o3:o2:o1:o0) using cascaded full adders. The result
is a 4-bit neighbour count per bit position, computed entirely with bitwise operations.

### Left/right shifting across word boundaries

The left and right neighbours of cell `c[w]` require bits from the adjacent words
`c[w-1]` and `c[w+1]`:

```cpp
cL = (cC << 1) | (c[w-1] >> 63);   // left neighbour
cR = (cC >> 1) | (c[w+1] << 63);   // right neighbour
```

The shift moves all 64 cells one position. The bit that "falls off" one word is
replaced by the bit that falls off the adjacent word. This is the bit-packed equivalent
of the SIMD engine's `load(col - 1)` and `load(col + 1)`.

### GoL rule — bitwise selection

With the 4-bit count (o3:o2:o1:o0) available per bit position, the rule is:

```cpp
is3 = ~o3 & ~o2 &  o1 & o0;         // count == 3 (binary 0011)
is4 = ~o3 &  o2 & ~o1 & ~o0;        // count == 4 (binary 0100, i.e. 3 neighbours + self)
result = is3 | (cC & is4);
```

Wait — why count == 4? Because `sum9` adds all 9 cells (8 neighbours + the cell
itself). A living cell with 2 neighbours contributes count = 3 (self + 2), but
`rowSum3` for the current row includes the cell itself. The actual convention in this
implementation: `c1:c0` is computed from `cL, cC, cR` which includes the centre cell.
So the total 9-cell sum equals neighbours + self. Birth occurs when total == 3 (dead
cell, 3 neighbours). Survival occurs when total == 3 (alive, 2 neighbours) or total ==
4 (alive, 3 neighbours). The expression `is3 | (cC & is4)` captures both: `is3`
handles birth and 2-neighbour survival; `cC & is4` handles 3-neighbour survival.

*Correction:* The implementation actually computes `rowSum3(cL, cC, cR)` which sums
the centre cell and its two horizontal neighbours. The centre cell is included in
`sum9`. So `sum9 == neighbours + alive`. For a dead cell: sum9 == neighbours. For an
alive cell: sum9 == neighbours + 1. Birth: dead & sum9 == 3 → `is3 & ~cC` (but since
cC==0, `is3` suffices). Survival: alive & sum9 ∈ {3,4} → `cC & (is3 | is4)`. The code
`is3 | (cC & is4)` is equivalent.

### Why it is faster

1. **64x data-level parallelism.** Each bitwise instruction processes 64 cells.
2. **8x less memory.** The working set fits in cache more readily. Three rows of a
   50,000-column grid occupy 3 x 782 x 8 = 18.7 KB in bit-packed form, versus 150 KB
   as bytes. The entire 3-row working set fits in L1 cache (typically 32-48 KB).
3. **8x less memory bandwidth.** Fewer bytes transferred between DRAM and cache.
4. **No branches.** The entire computation is bitwise arithmetic.
5. **Few instructions.** The 9-cell sum and rule require approximately 25 bitwise
   operations per 64 cells, compared to 8 additions + 2 comparisons per 1 cell in the
   SIMD engine.

---

## 4. CUDA byte kernel (GPU)

**File:** `gol_cuda.cu` — `golKernel`

### Cell representation

Same as the CPU byte engines: one `uint8_t` per cell, flat row-major. The grid is
copied to GPU global memory (DRAM) before the simulation begins.

### The GPU execution model

A GPU has many streaming multiprocessors (SMs), each running thousands of threads
grouped into blocks. Threads within a block execute in lockstep groups of 32 called
warps. All 32 threads in a warp execute the same instruction simultaneously on
different data.

Each SM has a small, fast on-chip memory called **shared memory** (analogous to an
explicitly managed L1 cache). Shared memory is accessible only to threads within the
same block. It is roughly 100x faster than global memory (DRAM).

### Block and tile dimensions

Each block is 32 x 8 = 256 threads. Each thread processes 4 consecutive cells, so a
block covers 128 columns x 8 rows = 1,024 cells. The shared memory tile includes a
1-cell halo on all sides: (128 + 2) x (8 + 2) = 130 x 10 = 1,300 bytes.

### Phase 1: cooperative tile load

All 256 threads in the block collaborate to load the tile from global memory into shared
memory. The interior (128 x 10 cells) is loaded as 32 `uint32_t` words per tile row:

```cpp
uint32_t packed = *reinterpret_cast<const uint32_t*>(&src[gr * cols + gc]);
tile[tileBase + 0] = packed;
tile[tileBase + 1] = packed >> 8;
tile[tileBase + 2] = packed >> 16;
tile[tileBase + 3] = packed >> 24;
```

Each thread loads one `uint32_t` (4 consecutive bytes) from global memory, then unpacks
the 4 bytes into 4 individual `uint8_t` entries in the shared memory tile.

**Why `uint32_t` loads?** GPU memory transactions operate on 32-byte sectors. When 32
threads in a warp each load one `uint32_t` at consecutive addresses, the warp reads
32 x 4 = 128 contiguous bytes, spanning 4 sectors. This is **coalesced** — every byte
in each sector is used. Without vectorised loads, each thread loading a single byte
would still trigger a full 32-byte sector read, wasting 31 of 32 bytes.

**Why unpack into `uint8_t` tile?** The shared memory tile uses `uint8_t` entries
rather than `uint32_t` to avoid bank conflicts. Shared memory is divided into 32 banks,
each 4 bytes wide. In the compute phase, each thread accesses cells at stride 4 in the
tile (because each thread owns 4 consecutive cells). With `uint32_t` entries, stride-4
access maps to `(4 * threadIdx.x) % 32`, hitting only 8 of 32 banks — a 4-way bank
conflict that serialises accesses. With `uint8_t` entries, the same stride-4 access
maps to `floor(4 * threadIdx.x / 4) % 32 = threadIdx.x % 32`, hitting all 32 banks —
zero conflicts.

The left and right halo columns (1 cell each, 10 rows) are loaded separately by 20
threads using scalar byte loads.

After loading, all threads synchronise with `__syncthreads()` to ensure the tile is
fully populated before any thread reads from it.

### Phase 2: compute

Each thread computes the GoL rule for its 4 cells using the shared memory tile:

```cpp
for (int k = 0; k < 4; k++) {
    unsigned int cx = tx + k;
    unsigned int nb = tile[(ty-1)*tileW + cx-1] + tile[(ty-1)*tileW + cx] + ...
    unsigned int alive = tile[ty * tileW + cx];
    unsigned int cell = (nb == 3) | (alive & (nb == 2));
    result |= (cell & 0xFF) << (k * 8);
}
```

The 8 neighbour values are read from shared memory (fast, ~1 cycle per read with no
bank conflicts). The GoL rule is branch-free, identical to the CPU SIMD version. The
4 output cells are packed into a `uint32_t` and written to global memory with a single
4-byte store — again coalesced across the warp.

### Why shared memory is necessary

Without shared memory, each thread would read its 8 neighbours directly from global
memory. For a block of 1,024 cells, that is 9,216 global reads. With shared memory,
the block performs 1,300 global reads (to fill the tile), then 9,216 shared memory reads
(fast). The global memory traffic is reduced by 7x.

### Edge handling

When the grid width is not a multiple of 128, the rightmost block has fewer valid
columns. Threads whose cells extend beyond the grid boundary fall back to a scalar
per-byte path. This affects at most one block column out of hundreds — negligible
impact on overall performance.

---

## 5. CUDA bit-packed kernel (GPU)

**File:** `gol_cuda.cu` — `golBitPackKernel`

### Cell representation

Same as the CPU bit-packed engine: 64 cells per `uint64_t`, stored as an array of
words. The grid is copied to GPU global memory as `uint64_t` values.

### Block and tile structure

Each block is 32 x 8 = 256 threads. Each thread processes one `uint64_t` word (64
cells). A block therefore covers 32 x 64 = 2,048 columns x 8 rows = 16,384 cells.

The shared memory tile is (32 + 2) x (8 + 2) = 340 `uint64_t` values = 2,720 bytes.
Each thread loads one word, and the tile includes a 1-word halo for cross-word bit
shifting.

### Loading

The cooperative load is straightforward: each thread loads one `uint64_t` from global
memory into the corresponding tile position. With 32 threads per warp each loading 8
bytes at consecutive addresses, the warp reads 256 contiguous bytes — well coalesced.

### Bit shifting across word boundaries

The left and right neighbours require shifting bits by one position. Bits that shift out
of one word must come from the adjacent word:

```cpp
cL = (cC << 1) | (stile[ty * tileW + tx - 1] >> 63);
cR = (cC >> 1) | (stile[ty * tileW + tx + 1] << 63);
```

The adjacent words are available in shared memory (the halo), so this cross-word
shifting is a fast local operation.

### Computation

The `d_rowSum3` and `d_sum9` functions are identical to the CPU versions. The binary
adder circuit computes the per-bit neighbour count, and the bitwise rule selects birth
and survival. Each thread performs approximately 25 bitwise operations to process 64
cells.

### Why it is the fastest

1. **64x data-level parallelism per thread.** Each bitwise instruction operates on 64
   cells. Combined with 256 threads per block and thousands of blocks, the GPU
   processes billions of cells per kernel launch.
2. **Minimal memory traffic.** The 50,000 x 50,000 grid is 313 MB in bit-packed form.
   Each generation reads and writes this once, requiring 626 MB of bandwidth. The V100
   provides 900 GB/s of memory bandwidth, so the theoretical minimum time per
   generation is 0.7 ms. The measured time of 1.2 ms per generation indicates the
   kernel runs at ~52% of peak bandwidth — reasonable given the compute and
   synchronisation overhead.
3. **Excellent coalescing.** Each warp loads 32 consecutive `uint64_t` words = 256
   bytes = 8 sectors, all fully utilised.
4. **Small shared memory footprint.** 2,720 bytes per block allows high occupancy
   (many blocks per SM), keeping the GPU's execution units busy.

---

## Summary: what each optimisation targets

| Technique | What it improves | Where applied |
|-----------|-----------------|---------------|
| OpenMP `parallel for` | Thread-level parallelism (multiple cores) | All CPU engines |
| Branch-free GoL rule | Eliminates branch mispredictions | SIMD, bit-packed, both CUDA |
| SIMD vector instructions | Data-level parallelism (16-64 cells/instruction) | SIMD engine |
| Bit-packing | 8x less memory, 64 cells per operation | Bit-packed CPU and CUDA |
| Bitwise adder circuits | Neighbour counting without per-cell arithmetic | Bit-packed CPU and CUDA |
| Compile-time templates | Eliminates boundary branches from inner loop | SIMD engine |
| GPU shared memory tiling | Reduces global memory traffic, enables data reuse | Both CUDA engines |
| Coalesced global loads | Maximises memory bandwidth utilisation | Both CUDA engines |
| `uint32_t` vectorised loads | Amortises alignment overhead across 4 bytes | CUDA byte engine |
| `uint8_t` shared memory tile | Eliminates bank conflicts (stride-4 → stride-1 in banks) | CUDA byte engine |
