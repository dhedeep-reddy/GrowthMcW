#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void reduceSum(int* input, int* output)
{
    __shared__ int sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = input[i];

    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    const int N = 1<<20;
    int size = N * sizeof(int);

    int *h_input = new int[N];
    int *h_output = new int[4096];

    for(int i = 0; i < N; i++)
        h_input[i] = 1;

    int *d_input, *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, 4096 * sizeof(int));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    reduceSum<<<4096,256>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, 4096 * sizeof(int), cudaMemcpyDeviceToHost);

    int finalSum = 0;
    for(int i = 0; i < 4096; i++)
        finalSum += h_output[i];

    cout << "Sum = " << finalSum << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}