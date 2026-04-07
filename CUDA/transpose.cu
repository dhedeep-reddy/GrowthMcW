#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void transposeNoConflict(int *input, int *output, int N)
{
    __shared__ int tile[TILE_SIZE][TILE_SIZE + 1];  // ✅ Padding added

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load (coalesced)
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];

    __syncthreads();

    // Transpose indices
    int tx = blockIdx.y * TILE_SIZE + threadIdx.x;
    int ty = blockIdx.x * TILE_SIZE + threadIdx.y;

    // ✅ No bank conflict
    if (tx < N && ty < N)
        output[ty * N + tx] = tile[threadIdx.x][threadIdx.y];
}

int main()
{
    const int N = 1024;
    int size = N * N * sizeof(int);

    int *h_in = new int[N * N];
    int *h_out = new int[N * N];

    for(int i = 0; i < N * N; i++)
        h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1)/TILE_SIZE,
                (N + TILE_SIZE - 1)/TILE_SIZE);

    transposeNoConflict<<<blocks, threads>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "Optimized Transpose Done\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}