#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_SIZE 32

using std::vector;
using std::cout;

__global__ void tiledMatrixMul(const float *a, const float *b, float *c, int N) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        if (i * TILE_SIZE + tx < N && row < N) 
            tile_a[ty][tx] = a[row * N + i * TILE_SIZE + tx];
        else
            tile_a[ty][tx] = 0.0;

        if (i * TILE_SIZE + ty < N && col < N) 
            tile_b[ty][tx] = b[(i * TILE_SIZE + ty) * N + col];
        else
            tile_b[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_a[ty][k] * tile_b[k][tx];

        __syncthreads();
    }
    if (row < N && col < N)
        c[row * N + col] = sum;
}

int main() {
    int N = 4096; // Matrix size
    size_t bytes = N * N * sizeof(float);

    vector<float> h_a(N * N), h_b(N * N), h_c(N * N);
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    tiledMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time elapsed: " << milliseconds << " ms\n";

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
