#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
using namespace std;

#define N 1024
#define BLOCK_SIZE 16
#define SPARSITY 0.9f

struct COOMatrix {
    float* values;   
    int*   rowIdx;   
    int*   colIdx;   
    int    nnz;      
};

void generate_sparse_coo(float* dense, COOMatrix& coo)
{
    srand(42);

    int nnz = 0;
    for (int i = 0; i < N * N; i++) {
        float r = (float)rand() / RAND_MAX;
        if (r >= SPARSITY) {
            dense[i] = (float)(rand() % 9 + 1);  
            nnz++;
        } else {
            dense[i] = 0.0f;
        }
    }

    // Allocate COO arrays
    coo.nnz    = nnz;
    coo.values = new float[nnz];
    coo.rowIdx = new int[nnz];
    coo.colIdx = new int[nnz];

    int idx = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (dense[i * N + j] != 0.0f) {
                coo.values[idx] = dense[i * N + j];
                coo.rowIdx[idx] = i;
                coo.colIdx[idx] = j;
                idx++;
            }
        }
    }
}


void cpu_coo_matmul(const COOMatrix& A, float* B, float* C)
{
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;

    for (int p = 0; p < A.nnz; p++) {
        int   r   = A.rowIdx[p];
        int   k   = A.colIdx[p];
        float val = A.values[p];

        // Scatter: C[r, :] += val * B[k, :]
        for (int j = 0; j < N; j++)
            C[r * N + j] += val * B[k * N + j];
    }
}

__global__ void gpu_coo_scatter(
    const float* __restrict__ vals,
    const int*   __restrict__ rowIdx,
    const int*   __restrict__ colIdx,
    const float* __restrict__ B,
    float*                    C,
    int nnz)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;

    int   r   = rowIdx[p];
    int   k   = colIdx[p];
    float val = vals[p];

    for (int j = 0; j < N; j++)
        atomicAdd(&C[r * N + j], val * B[k * N + j]);
}

#define TX 32
#define TY 16

__global__ void gpu_coo_no_atomic(
    const float* __restrict__ vals,
    const int*   __restrict__ rowIdx,
    const int*   __restrict__ colIdx,
    const float* __restrict__ B,
    float*                    C,
    int nnz)
{
    int p   = blockIdx.y * blockDim.y + threadIdx.y;   // non-zero index
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // output column

    if (p >= nnz || col >= N) return;

    int   r   = rowIdx[p];
    int   k   = colIdx[p];
    float val = vals[p];

    atomicAdd(&C[r * N + col], val * B[k * N + col]);
}

bool verify(float* cpu, float* gpu)
{
    for (int i = 0; i < N * N; i++)
        if (fabs(cpu[i] - gpu[i]) > 1e-2f) {
            printf("Mismatch at %d: cpu=%.4f gpu=%.4f\n", i, cpu[i], gpu[i]);
            return false;
        }
    return true;
}

int main()
{
    size_t dense_size = (size_t)N * N * sizeof(float);

    float* h_A     = new float[N * N];
    float* h_B     = new float[N * N];
    float* h_C_cpu = new float[N * N]();
    float* h_C_gpu = new float[N * N]();

    COOMatrix A_coo;
    generate_sparse_coo(h_A, A_coo);

    srand(123);
    for (int i = 0; i < N * N; i++) {
        float r = (float)rand() / RAND_MAX;
        h_B[i]  = (r >= SPARSITY) ? (float)(rand() % 9 + 1) : 0.0f;
    }

    printf("Matrix size      : %d x %d\n", N, N);
    printf("Sparsity         : %.0f%%\n",   SPARSITY * 100);

    cpu_coo_matmul(A_coo, h_B, h_C_cpu);

    float *d_vals, *d_B, *d_C;
    int   *d_rowIdx, *d_colIdx;

    cudaMalloc(&d_vals,   A_coo.nnz * sizeof(float));
    cudaMalloc(&d_rowIdx, A_coo.nnz * sizeof(int));
    cudaMalloc(&d_colIdx, A_coo.nnz * sizeof(int));
    cudaMalloc(&d_B,      dense_size);
    cudaMalloc(&d_C,      dense_size);

    cudaMemcpy(d_vals,   A_coo.values, A_coo.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowIdx, A_coo.rowIdx, A_coo.nnz * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, A_coo.colIdx, A_coo.nnz * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,      h_B,          dense_size,                 cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, dense_size);

    dim3 threads2(TX, TY);
    dim3 blocks2(
        (N + TX - 1) / TX,
        (A_coo.nnz  + TY - 1) / TY
    );

    gpu_coo_no_atomic<<<blocks2, threads2>>>(
        d_vals, d_rowIdx, d_colIdx, d_B, d_C, A_coo.nnz);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu, d_C, dense_size, cudaMemcpyDeviceToHost);
    printf("Kernel (per-col) : %s\n", verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong");

    cudaMemset(d_C, 0, dense_size);

    int flat_threads = 256;
    int flat_blocks  = (A_coo.nnz + flat_threads - 1) / flat_threads;

    gpu_coo_scatter<<<flat_blocks, flat_threads>>>(
        d_vals, d_rowIdx, d_colIdx, d_B, d_C, A_coo.nnz);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu, d_C, dense_size, cudaMemcpyDeviceToHost);
    printf("Kernel (scatter) : %s\n", verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong");

    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;
    delete[] A_coo.values; delete[] A_coo.rowIdx; delete[] A_coo.colIdx;
    cudaFree(d_vals); cudaFree(d_rowIdx); cudaFree(d_colIdx);
    cudaFree(d_B);    cudaFree(d_C);

    return 0;
}