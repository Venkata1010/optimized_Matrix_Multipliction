# Optimized Matrix Multiplication on GPU Using Tiling and Shared Memory

This project implements two CUDA kernels for square matrix–matrix multiplication and evaluates their performance on an NVIDIA GPU. The **baseline** kernel performs a straightforward global-memory compute, and the **optimized** kernel applies a tiling strategy with shared memory to increase data reuse and reduce global memory traffic, following the methodology described in the final project report.

## Source files

- `naive.cu` — baseline CUDA implementation  
  - Kernel identifier in code: `naiveMatrixMul`  
  - Term used in the report: `matrixMultNaive`

- `opt.cu` — optimized CUDA implementation using tiling and shared memory  
  - Kernel identifier in code: `tiledMatrixMul`  
  - Term used in the report: `matrixMultOpt`  
  - Uses a compile-time tile macro (`TILE_SIZE`) defined in the source

## Method

**Baseline (naive) kernel — `naiveMatrixMul` / `matrixMultNaive`.**  
Each thread computes a single element of the output matrix `C[row, col]` by iterating over `k` and accumulating `A[row, k] * B[k, col]` with data fetched from global memory.

**Optimized (tiled) kernel — `tiledMatrixMul` / `matrixMultOpt`.**  
Computation is decomposed into square tiles of size `TILE_SIZE × TILE_SIZE`. For each tile phase, cooperative threads load sub-blocks of `A` and `B` into `__shared__` memory, synchronize, and perform the inner products using the cached tiles before advancing to the next phase. This tiling scheme reduces global memory transactions and improves arithmetic intensity.

## Program behavior

- Allocates host and device buffers for `A`, `B`, and `C`.  
- Configures a 2-D grid of thread blocks consistent with the kernel’s mapping from threads to output elements.  
- Executes the selected kernel and reports runtime and performance metrics as implemented in the source and described in the report.

## Assumptions and notes

- Matrices are square (`N × N`).  
- The optimized kernel uses a compile-time `TILE_SIZE` macro defined in `opt.cu`. Choose matrix sizes and launch configurations consistent with that tiling scheme.  
- Terminology and identifiers in this README follow the exact names used in the code and in the report:
  - Baseline: `naiveMatrixMul` (code) / `matrixMultNaive` (report)  
  - Optimized: `tiledMatrixMul` (code) / `matrixMultOpt` (report)

## Environment

- NVIDIA GPU with a compatible CUDA driver  
- CUDA Toolkit (for `nvcc`, CUDA runtime headers, and libraries)

## Documentation

For the full description of the approach, analysis, and results, see the final project report.
