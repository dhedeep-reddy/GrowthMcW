#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixAdd(int* A, int* B, int* C, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width)
    {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main()
{
    const int width = 4;
    const int height = 4;
    int size = width * height * sizeof(int);

    int *h_A = new int[width * height];
    int *h_B = new int[width * height];
    int *h_C = new int[width * height];

    for(int i = 0; i < width * height; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    matrixAdd<<<blocks, threads>>>(d_A, d_B, d_C, width, height);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result matrix:" << endl;

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            cout << h_C[i * width + j] << " ";
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