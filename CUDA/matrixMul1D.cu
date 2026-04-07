#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMul1D(int* A, int* B, int* C,
                           int rowsA, int colsA, int colsB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = rowsA * colsB;

    if (idx < totalElements)
    {
        // Convert 1D index → 2D (row, col)
        int row = idx / colsB;
        int col = idx % colsB;

        int sum = 0;

        for (int k = 0; k < colsA; k++)
        {
            sum += A[row * colsA + k] *
                   B[k * colsB + col];
        }

        C[row * colsB + col] = sum;
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

    // 1D thread configuration
    int totalThreads = rowsA * colsB;
    int threadsPerBlock = 256;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    matrixMul1D<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C,
                                             rowsA, colsA, colsB);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cout << "Result Matrix (first 5x5 block):\n";

    // Print only small part (avoid huge output)
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            cout << h_C[i * colsB + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}