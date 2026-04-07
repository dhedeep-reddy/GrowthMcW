#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorAdd(int *A, int *B, int *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    int N = 1024;
    int size = N * sizeof(int);

    int *A, *B, *C;

    // Unified Memory allocation
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch kernel
    vectorAdd<<<blocks, threads>>>(A, B, C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    cout << "Result sample:" << endl;

    for (int i = 0; i < 10; i++)
        cout << C[i] << " ";

    cout << endl;

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}