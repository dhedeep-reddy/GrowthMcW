#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16
#define WIDTH 4   // Vector width

// Vectorized tiled matrix multiplication
__global__ void matrixMulVectorized(int4* A, int4* B, int4* C,
                                    int rowsA, int colsA, int colsB)
{
    __shared__ int4 tileA[TILE_SIZE][TILE_SIZE / WIDTH];
    __shared__ int4 tileB[TILE_SIZE][TILE_SIZE / WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int globalRow = (blockIdx.y * TILE_SIZE + ty);   // row index
    int globalCol = (blockIdx.x * TILE_SIZE + tx);   // col index

    // Because of int4, columns are divided by WIDTH
    int vecCol = globalCol / WIDTH;

    int4 sum = make_int4(0, 0, 0, 0);

    for (int t = 0; t < (colsA + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Load tile A (vectorized)
        int tiledCol = t * TILE_SIZE + tx;

        if (globalRow < rowsA && tiledCol < colsA / WIDTH)
            tileA[ty][tx] = A[globalRow * (colsA / WIDTH) + tiledCol];
        else
            tileA[ty][tx] = make_int4(0, 0, 0, 0);

        // Load tile B (vectorized)
        int tiledRow = t * TILE_SIZE + ty;

        if (globalCol < colsB && tiledRow < colsA / WIDTH)
            tileB[ty][tx] = B[tiledRow * (colsB / WIDTH) + vecCol];
        else
            tileB[ty][tx] = make_int4(0, 0, 0, 0);

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_SIZE / WIDTH; k++)
        {
            int4 a = tileA[ty][k];
            int4 b = tileB[k][tx];

            // Manually unrolled multiply (like OpenCL)
            sum.x += a.x * b.x;
            sum.y += a.y * b.y;
            sum.z += a.z * b.z;
            sum.w += a.w * b.w;
        }

        __syncthreads();
    }

    // Store result
    if (globalRow < rowsA && vecCol < colsB / WIDTH)
    {
        C[globalRow * (colsB / WIDTH) + vecCol] = sum;
    }
}

int main()
{
    const int N = 1024;

    int size = N * N * sizeof(int);

    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];

    for (int i = 0; i < N * N; i++)
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

    // Cast to int4
    int4* d_A_vec = reinterpret_cast<int4*>(d_A);
    int4* d_B_vec = reinterpret_cast<int4*>(d_B);
    int4* d_C_vec = reinterpret_cast<int4*>(d_C);

    dim3 threads(TILE_SIZE / WIDTH, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulVectorized<<<blocks, threads>>>(
        d_A_vec, d_B_vec, d_C_vec,
        N, N, N
    );

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result Matrix:" << endl;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
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