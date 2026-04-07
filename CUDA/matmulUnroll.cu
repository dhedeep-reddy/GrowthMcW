#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16

__global__ void matrixMulTiled(int* A, int* B, int* C,
                               int rowsA, int colsA, int colsB)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    for(int t = 0; t < (colsA + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Load tile A
        if(row < rowsA && (t * TILE_SIZE + threadIdx.x) < colsA)
            tileA[threadIdx.y][threadIdx.x] =
                A[row * colsA + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        // Load tile B
        if(col < colsB && (t * TILE_SIZE + threadIdx.y) < colsA)
            tileB[threadIdx.y][threadIdx.x] =
                B[(t * TILE_SIZE + threadIdx.y) * colsB + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // 🔥 Loop Unrolling applied here
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < rowsA && col < colsB)
        C[row * colsB + col] = sum;
}

int main()
{
    const int N = 1024;

    int size = N * N * sizeof(int);

    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];

    for(int i = 0; i < N * N; i++)
    {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N, N, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result Matrix:" << endl;

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout << h_C[i * N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}