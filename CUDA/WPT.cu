#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16
#define WPT 4   // Work Per Thread

// MPT Kernel
__global__ void matrixMulMPT(int* A, int* B, int* C,
                            int rowsA, int colsA, int colsB)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE * WPT];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE * WPT + threadIdx.x;

    int sum[WPT] = {0};

    for (int t = 0; t < colsA / TILE_SIZE; t++)
    {
        // Load A tile
        if (row < rowsA && (t * TILE_SIZE + threadIdx.x) < colsA)
        {
            tileA[threadIdx.y][threadIdx.x] =
                A[row * colsA + t * TILE_SIZE + threadIdx.x];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tiles (multiple columns)
        for (int w = 0; w < WPT; w++)
        {
            int bCol = col + w * TILE_SIZE;

            if ((t * TILE_SIZE + threadIdx.y) < colsA && bCol < colsB)
            {
                tileB[threadIdx.y][threadIdx.x + w * TILE_SIZE] =
                    B[(t * TILE_SIZE + threadIdx.y) * colsB + bCol];
            }
            else
            {
                tileB[threadIdx.y][threadIdx.x + w * TILE_SIZE] = 0;
            }
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_SIZE; k++)
        {
            int aVal = tileA[threadIdx.y][k];

            for (int w = 0; w < WPT; w++)
            {
                sum[w] += aVal *
                          tileB[k][threadIdx.x + w * TILE_SIZE];
            }
        }

        __syncthreads();
    }

    // Store results
    for (int w = 0; w < WPT; w++)
    {
        int outCol = col + w * TILE_SIZE;

        if (row < rowsA && outCol < colsB)
        {
            C[row * colsB + outCol] = sum[w];
        }
    }
}

int main()
{
    const int rowsA = 1024;
    const int colsA = 1024;
    const int colsB = 1024;

    int sizeA = rowsA * colsA * sizeof(int);
    int sizeB = colsA * colsB * sizeof(int);
    int sizeC = rowsA * colsB * sizeof(int);

    int *h_A = new int[rowsA * colsA];
    int *h_B = new int[colsA * colsB];
    int *h_C = new int[rowsA * colsB];

    // Initialize matrices
    for (int i = 0; i < rowsA * colsA; i++)
        h_A[i] = 1;

    for (int i = 0; i < colsA * colsB; i++)
        h_B[i] = 2;

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 threads(TILE_SIZE, TILE_SIZE);

    dim3 blocks((colsB + TILE_SIZE * WPT - 1) / (TILE_SIZE * WPT),
                (rowsA + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulMPT<<<blocks, threads>>>(d_A, d_B, d_C,
                                     rowsA, colsA, colsB);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cout << "Result Matrix (first 5x5):\n";

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            cout << h_C[i * colsB + j] << " ";
        }
        cout << endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}