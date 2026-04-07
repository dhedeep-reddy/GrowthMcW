#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMul(int* A, int* B, int* C,
                          int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rowsA && col < colsB)
    {
        int sum = 0;

        for(int k = 0; k < colsA; k++)
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

    for(int i = 0; i < rowsA * colsA; i++)
        h_A[i] = 1;

    for(int i = 0; i < colsA * colsB; i++)
        h_B[i] = 2;

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((colsB + 15)/16, (rowsA + 15)/16);

    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C,
                                   rowsA, colsA, colsB);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cout << "Result Matrix:" << endl;

    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            cout << h_C[i * colsB + j] << " ";
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