// register blocking, double buffering,WPT
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

#define N 1024
#define TILE_SIZE 16
#define WPT 2
#define RTS (TILE_SIZE / WPT)

void cpu_matmul(float* A, float* B, float* C)
{
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            float sum = 0;
            for(int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

__global__ void gpu_matmul_optimized(float* __restrict__ A,
                                     float* __restrict__ B,
                                     float* __restrict__ C)
{
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row    = blockIdx.y * TILE_SIZE + ty;
    int colBase = blockIdx.x * TILE_SIZE + tx * WPT;

    float sum[WPT] = {0.0f};
    int numTiles = N / TILE_SIZE;
    int ping = 0;

    #pragma unroll
    for(int w = 0; w < WPT; w++)
        As[ping][ty][tx * WPT + w] = A[row * N + (tx * WPT + w)];

    #pragma unroll
    for(int w = 0; w < WPT; w++)
        Bs[ping][ty][tx * WPT + w] = B[ty * N + colBase + w];

    __syncthreads();

    for(int t = 0; t < numTiles; t++)
    {
        int pong = 1 - ping;

        if(t + 1 < numTiles)
        {
            int next = t + 1;

            #pragma unroll
            for(int w = 0; w < WPT; w++)
                As[pong][ty][tx * WPT + w] =
                    A[row * N + (next * TILE_SIZE + tx * WPT + w)];

            #pragma unroll
            for(int w = 0; w < WPT; w++)
                Bs[pong][ty][tx * WPT + w] =
                    B[(next * TILE_SIZE + ty) * N + colBase + w];
        }

        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++)
        {
            float a_val = As[ping][ty][k];

            #pragma unroll
            for(int w = 0; w < WPT; w++)
                sum[w] += a_val * Bs[ping][k][tx * WPT + w];
        }

        __syncthreads();
        ping = pong;
    }

    #pragma unroll
    for(int w = 0; w < WPT; w++)
        if(row < N && colBase + w < N)
            C[row * N + colBase + w] = sum[w];
}

bool verify(float* cpu, float* gpu)
{
    for(int i = 0; i < N*N; i++)
        if(abs(cpu[i] - gpu[i]) > 1e-3)
            return false;
    return true;
}

int main()
{
    size_t size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    h_A     = (float*)malloc(size);
    h_B     = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    for(int i = 0; i < N*N; i++) { h_A[i] = rand() % 5; h_B[i] = rand() % 5; }

    cpu_matmul(h_A, h_B, h_C_cpu);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(RTS, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    gpu_matmul_optimized<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cout << (verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong") << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}