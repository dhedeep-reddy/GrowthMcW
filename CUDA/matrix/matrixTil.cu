#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

#define N 1024
#define TILE_SIZE 16

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

__global__ void gpu_matmul_Tilled(float* A, float* B, float* C)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for(int t = 0; t < N / TILE_SIZE; t++)
    {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        #pragma unroll 
        for(int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if(row < N && col < N)
        C[row * N + col] = sum;
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    gpu_matmul_Tilled<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cout << (verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong") << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}