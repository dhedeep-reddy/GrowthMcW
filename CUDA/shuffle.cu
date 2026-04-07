#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// 🔹 Warp-level reduction using shuffle
__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 🔹 Block-level reduction kernel
__global__ void reduceKernel(float* input, float* output, int N) {

    __shared__ float shared[BLOCK_SIZE / 32]; // one per warp

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < N) ? input[idx] : 0.0f;

    // Step 1: Warp-level reduction
    val = warpReduceSum(val);

    // Step 2: Write warp results to shared memory
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (lane == 0) {
        shared[warpId] = val;
    }

    __syncthreads();

    // Step 3: Final reduction (first warp only)
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (warpId == 0) {
        val = warpReduceSum(val);
    }

    // Step 4: Write result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}

// 🔹 Host code
int main() {
    int N = 1 << 20; // ~1M elements
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    float* h_output;

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // expected sum = N
    }

    // Device memory
    float *d_input, *d_output;

    cudaMalloc(&d_input, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_output, numBlocks * sizeof(float));

    // Launch kernel
    reduceKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    // Copy partial results back
    h_output = new float[numBlocks];
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float final_sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        final_sum += h_output[i];
    }

    std::cout << "Final Sum = " << final_sum << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}