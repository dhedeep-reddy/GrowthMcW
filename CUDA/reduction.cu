#include <iostream>
#include <cuda_runtime.h>

#define N (1 << 20)   // 1M elements (important for GPU utilization)
#define BLOCK_SIZE 256

// ==========================
// Warp Reduce
// ==========================
__inline__ __device__
int warpReduceSum(int val)
{
    for(int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ==========================
// Kernel: Each block reduces chunk
// ==========================
__global__ void reductionMultiBlock(int *input, int *partial, int n)
{
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int val = 0;

    // 🔥 Grid-stride loop (VERY IMPORTANT)
    while(idx < n)
    {
        val += input[idx];
        idx += blockDim.x * gridDim.x;
    }

    // Warp-level reduction
    val = warpReduceSum(val);

    // Store warp results
    if(tid % 32 == 0)
        sdata[tid / 32] = val;

    __syncthreads();

    // Final reduction in block
    if(tid < 32)
    {
        val = (tid < blockDim.x / 32) ? sdata[tid] : 0;
        val = warpReduceSum(val);

        if(tid == 0)
            partial[blockIdx.x] = val;
    }
}

// ==========================
// MAIN
// ==========================
int main()
{
    int *h_input = new int[N];
    for(int i = 0; i < N; i++)
        h_input[i] = 1;

    int *d_input, *d_partial;

    cudaMalloc(&d_input, N * sizeof(int));

    int numBlocks = 256;  // use many blocks
    cudaMalloc(&d_partial, numBlocks * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    reductionMultiBlock<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partial, N);

    // Copy partial sums back
    int *h_partial = new int[numBlocks];
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    int final_sum = 0;
    for(int i = 0; i < numBlocks; i++)
        final_sum += h_partial[i];

    std::cout << "Final Sum: " << final_sum << std::endl;

    cudaFree(d_input);
    cudaFree(d_partial);
    delete[] h_input;
    delete[] h_partial;

    return 0;
}