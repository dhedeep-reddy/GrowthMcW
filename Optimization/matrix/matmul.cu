#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

#define N 1024
#define BLOCK_SIZE 16

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

__global__ void gpu_matmul(float* A, float* B, float* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float sum = 0;
        for(int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];

        C[row*N + col] = sum;
    }
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
    size_t size = N*N*sizeof(float);

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    for(int i = 0; i < N*N; i++)
    {
        h_A[i] = rand() % 5;
        h_B[i] = rand() % 5;
    }

    cpu_matmul(h_A, h_B, h_C_cpu);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);

    gpu_matmul<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    if(verify(h_C_cpu, h_C_gpu))
        cout << "Correct" << endl;
    else
        cout << "Wrong" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}