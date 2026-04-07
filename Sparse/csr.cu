#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

#define N 1024
#define BLOCK_SIZE 16

#define SPARSITY 0.9f

struct CSRMatrix {
    float* values;      
    int*   colIdx;      
    int*   rowPtr;      
    int    nnz;         
};


void generate_sparse(float* dense, CSRMatrix& csr)
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

    csr.nnz    = nnz;
    csr.values = new float[nnz];
    csr.colIdx = new int[nnz];
    csr.rowPtr = new int[N + 1];

    int idx = 0;
    for (int i = 0; i < N; i++) {
        csr.rowPtr[i] = idx;
        for (int j = 0; j < N; j++) {
            if (dense[i * N + j] != 0.0f) {
                csr.values[idx] = dense[i * N + j];
                csr.colIdx[idx] = j;
                idx++;
            }
        }
    }
    csr.rowPtr[N] = nnz;
}

void cpu_sparse_matmul(const CSRMatrix& A_csr, float* B_dense, float* C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = A_csr.rowPtr[i]; p < A_csr.rowPtr[i + 1]; p++) {
                int k = A_csr.colIdx[p];
                sum += A_csr.values[p] * B_dense[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void gpu_sparse_matmul(
    const float* __restrict__ vals,
    const int*   __restrict__ colIdx,
    const int*   __restrict__ rowPtr,
    const float* __restrict__ B,
    float*                    C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        int start = rowPtr[row];
        int end   = rowPtr[row + 1];

        for (int p = start; p < end; p++) {
            sum += vals[p] * B[colIdx[p] * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool verify(float* cpu, float* gpu)
{
    for (int i = 0; i < N * N; i++)
        if (fabs(cpu[i] - gpu[i]) > 1e-2f)
            return false;
    return true;
}

int main()
{
    size_t dense_size = N * N * sizeof(float);

    float* h_A     = new float[N * N];
    float* h_B     = new float[N * N];
    float* h_C_cpu = new float[N * N]();
    float* h_C_gpu = new float[N * N]();

    CSRMatrix A_csr;
    generate_sparse(h_A, A_csr);   
    
    srand(123);
    for (int i = 0; i < N * N; i++) {
        float r = (float)rand() / RAND_MAX;
        h_B[i] = (r >= SPARSITY) ? (float)(rand() % 9 + 1) : 0.0f;
    }

    printf("Matrix size   : %d x %d\n", N, N);
    printf("Sparsity      : %.0f%%\n", SPARSITY * 100);

    cpu_sparse_matmul(A_csr, h_B, h_C_cpu);

    float *d_vals, *d_B, *d_C;
    int   *d_colIdx, *d_rowPtr;

    cudaMalloc(&d_vals,   A_csr.nnz * sizeof(float));
    cudaMalloc(&d_colIdx, A_csr.nnz * sizeof(int));
    cudaMalloc(&d_rowPtr, (N + 1)   * sizeof(int));
    cudaMalloc(&d_B,      dense_size);
    cudaMalloc(&d_C,      dense_size);

    cudaMemcpy(d_vals,   A_csr.values, A_csr.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, A_csr.colIdx, A_csr.nnz * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, A_csr.rowPtr, (N + 1)   * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,      h_B,          dense_size,                 cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gpu_sparse_matmul<<<blocks, threads>>>(d_vals, d_colIdx, d_rowPtr, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu, d_C, dense_size, cudaMemcpyDeviceToHost);

    cout << (verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong") << endl;

    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;
    delete[] A_csr.values; delete[] A_csr.colIdx; delete[] A_csr.rowPtr;
    cudaFree(d_vals); cudaFree(d_colIdx); cudaFree(d_rowPtr);
    cudaFree(d_B);    cudaFree(d_C);

    return 0;
}