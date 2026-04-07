#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorMul(int *A, int *B, int *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] * B[i];
}

int main()
{
    int N = 1024;
    int size = N * sizeof(int);

    // Host memory
    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    // Initialize host vectors
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i + 1;
    }

    // Device memory
    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch kernel
    vectorMul<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result from GPU to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result sample:" << endl;

    for (int i = 0; i < 10; i++)
        cout << h_C[i] << " ";

    cout << endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}