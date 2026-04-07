#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

#define N 1024
#define TILE_SIZE 16
#define WPT 4
#define RTS (TILE_SIZE / WPT)


#define CFLOAT4(ptr) (reinterpret_cast<const float4*>(ptr))

void cpu_matmul(float* A, float* B, float* C)
{
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            float sum = 0;
            for(int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

__global__ void gpu_matmul_optimized(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float*       __restrict__ C)
{
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 2];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row     = blockIdx.y * TILE_SIZE + ty;
    int colBase = blockIdx.x * TILE_SIZE + tx * WPT;

    float sum[WPT] = {0.0f};
    int numTiles = N / TILE_SIZE;
    int ping = 0;

    // Load first tile
    {
        float4 a4 = CFLOAT4(A)[row * (N / WPT) + (0 * RTS + tx)];
        As[ping][ty][tx*WPT+0] = a4.x;
        As[ping][ty][tx*WPT+1] = a4.y;
        As[ping][ty][tx*WPT+2] = a4.z;
        As[ping][ty][tx*WPT+3] = a4.w;

        float4 b4 = CFLOAT4(B)[(0 * TILE_SIZE + ty) * (N / WPT)
                               + (blockIdx.x * RTS + tx)];
        Bs[ping][ty][tx*WPT+0] = b4.x;
        Bs[ping][ty][tx*WPT+1] = b4.y;
        Bs[ping][ty][tx*WPT+2] = b4.z;
        Bs[ping][ty][tx*WPT+3] = b4.w;
    }
    __syncthreads();

    for(int t = 0; t < numTiles; t++)
    {
        int pong = 1 - ping;

        // Prefetch next tile
        if(t + 1 < numTiles)
        {
            int next = t + 1;

            float4 a4 = CFLOAT4(A)[row * (N / WPT) + (next * RTS + tx)];
            As[pong][ty][tx*WPT+0] = a4.x;
            As[pong][ty][tx*WPT+1] = a4.y;
            As[pong][ty][tx*WPT+2] = a4.z;
            As[pong][ty][tx*WPT+3] = a4.w;

            float4 b4 = CFLOAT4(B)[(next * TILE_SIZE + ty) * (N / WPT)
                                   + (blockIdx.x * RTS + tx)];
            Bs[pong][ty][tx*WPT+0] = b4.x;
            Bs[pong][ty][tx*WPT+1] = b4.y;
            Bs[pong][ty][tx*WPT+2] = b4.z;
            Bs[pong][ty][tx*WPT+3] = b4.w;
        }

        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++)
        {
            float a_val = As[ping][ty][k];
            #pragma unroll
            for(int w = 0; w < WPT; w++)
                sum[w] += a_val * Bs[ping][k][tx*WPT+w];
        }

        __syncthreads();
        ping = pong;
    }

    #pragma unroll
    for(int w = 0; w < WPT; w++)
        if(row < N && colBase + w < N)
            C[row * N + colBase + w] = sum[w];
}

bool verify(float* cpu, float* gpu)
{
    for(int i = 0; i < N*N; i++)
        if(abs(cpu[i] - gpu[i]) > 1e-2f)
        {
            printf("Mismatch at %d: cpu=%.4f gpu=%.4f\n", i, cpu[i], gpu[i]);
            return false;
        }
    return true;
}

int main()
{
    size_t size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    h_A     = (float*)malloc(size);
    h_B     = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    srand(42);
    for(int i = 0; i < N*N; i++) { 
        h_A[i] = rand() % 5; 
        h_B[i] = rand() % 5; 
    }

    cpu_matmul(h_A, h_B, h_C_cpu);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemset(d_C, 0, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(RTS, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    gpu_matmul_optimized<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cout << (verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong") << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}