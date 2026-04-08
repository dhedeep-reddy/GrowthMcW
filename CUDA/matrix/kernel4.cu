#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;


#define N 4096

#define TILE_M  32      
#define TILE_N  32      
#define TILE_K  64     
#define WPT      4      


#define RTS_N  (TILE_N / WPT)   
#define RTS_K  (TILE_K / WPT)   
static_assert(TILE_N % WPT == 0, "TILE_N must be divisible by WPT");
static_assert(TILE_K % WPT == 0, "TILE_K must be divisible by WPT");


#define CFLOAT4(ptr) (reinterpret_cast<const float4*>(ptr))

__global__ void gpu_matmul_optimized(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float*       __restrict__ C)
{
    
    __shared__ float As[2][TILE_M][TILE_K];
    __shared__ float Bs[2][TILE_K][TILE_N + 2];   
  

    const int tx = threadIdx.x;   
    const int ty = threadIdx.y;   

    
    const int row     = blockIdx.y * TILE_M + ty;          
    const int colBase = blockIdx.x * TILE_N + tx * WPT;   

    float sum[WPT] = {0.0f};

    const int numTiles = N / TILE_K;
    int ping = 0;

  
    {
        const int tid      = ty * RTS_N + tx;               
        const int nThreads = TILE_M * RTS_N;                
        const int nVec     = (TILE_M * TILE_K) / WPT;      

        for (int i = tid; i < nVec; i += nThreads) {
            int lrow = i / (TILE_K / WPT);                  
            int lcol = i % (TILE_K / WPT);                  

            int grow = blockIdx.y * TILE_M + lrow;
            int gk4  = 0 * (TILE_K / WPT) + lcol;          

            float4 a4 = CFLOAT4(A)[grow * (N / WPT) + gk4];
            As[ping][lrow][lcol*WPT+0] = a4.x;
            As[ping][lrow][lcol*WPT+1] = a4.y;
            As[ping][lrow][lcol*WPT+2] = a4.z;
            As[ping][lrow][lcol*WPT+3] = a4.w;
        }

        
        const int nVecB = (TILE_K * TILE_N) / WPT;

        for (int i = tid; i < nVecB; i += nThreads) {
            int lk   = i / (TILE_N / WPT);
            int lcol = i % (TILE_N / WPT);

            int gk   = 0 * TILE_K + lk;
            int gcol4= blockIdx.x * (TILE_N / WPT) + lcol;

            float4 b4 = CFLOAT4(B)[gk * (N / WPT) + gcol4];
            Bs[ping][lk][lcol*WPT+0] = b4.x;
            Bs[ping][lk][lcol*WPT+1] = b4.y;
            Bs[ping][lk][lcol*WPT+2] = b4.z;
            Bs[ping][lk][lcol*WPT+3] = b4.w;
        }
    }
    __syncthreads();

    for (int t = 0; t < numTiles; t++)
    {
        const int pong = 1 - ping;

        if (t + 1 < numTiles) {
            const int next     = t + 1;
            const int tid      = ty * RTS_N + tx;
            const int nThreads = TILE_M * RTS_N;

            // A
            const int nVecA = (TILE_M * TILE_K) / WPT;
            for (int i = tid; i < nVecA; i += nThreads) {
                int lrow = i / (TILE_K / WPT);
                int lcol = i % (TILE_K / WPT);
                int grow = blockIdx.y * TILE_M + lrow;
                int gk4  = next * (TILE_K / WPT) + lcol;

                float4 a4 = CFLOAT4(A)[grow * (N / WPT) + gk4];
                As[pong][lrow][lcol*WPT+0] = a4.x;
                As[pong][lrow][lcol*WPT+1] = a4.y;
                As[pong][lrow][lcol*WPT+2] = a4.z;
                As[pong][lrow][lcol*WPT+3] = a4.w;
            }

            // B
            const int nVecB = (TILE_K * TILE_N) / WPT;
            for (int i = tid; i < nVecB; i += nThreads) {
                int lk    = i / (TILE_N / WPT);
                int lcol  = i % (TILE_N / WPT);
                int gk    = next * TILE_K + lk;
                int gcol4 = blockIdx.x * (TILE_N / WPT) + lcol;

                float4 b4 = CFLOAT4(B)[gk * (N / WPT) + gcol4];
                Bs[pong][lk][lcol*WPT+0] = b4.x;
                Bs[pong][lk][lcol*WPT+1] = b4.y;
                Bs[pong][lk][lcol*WPT+2] = b4.z;
                Bs[pong][lk][lcol*WPT+3] = b4.w;
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[ping][ty][k];
            #pragma unroll
            for (int w = 0; w < WPT; w++)
                sum[w] += a_val * Bs[ping][k][tx * WPT + w];
        }

        __syncthreads();
        ping = pong;
    }

    #pragma unroll
    for (int w = 0; w < WPT; w++)
        if (row < N && colBase + w < N)
            C[row * N + colBase + w] = sum[w];
}

void cpu_matmul(float* A, float* B, float* C)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

bool verify(float* cpu, float* gpu)
{
    for (int i = 0; i < N*N; i++)
        if (fabs(cpu[i] - gpu[i]) > 1e-2f) {
            printf("Mismatch at %d: cpu=%.4f gpu=%.4f\n", i, cpu[i], gpu[i]);
            return false;
        }
    return true;
}

int main()
{
    size_t size = (size_t)N * N * sizeof(float);

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    h_A     = (float*)malloc(size);
    h_B     = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    srand(42);
    for (int i = 0; i < N*N; i++) {
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

    // Block: (RTS_N, TILE_M) = (TILE_N/WPT, TILE_M)
    dim3 threads(RTS_N, TILE_M);
    dim3 blocks(N / TILE_N, N / TILE_M);

    printf("Grid  : (%d, %d)\n", blocks.x,  blocks.y);
    printf("Block : (%d, %d) = %d threads\n",
           threads.x, threads.y, threads.x * threads.y);
    printf("Tile  : M=%d  N=%d  K=%d  WPT=%d\n", TILE_M, TILE_N, TILE_K, WPT);

    // Shared mem estimate
    size_t smem = 2 * (TILE_M * TILE_K + TILE_K * (TILE_N + 2)) * sizeof(float);
    printf("Shared mem/block: %.2f KB\n", smem / 1024.0f);

    gpu_matmul_optimized<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cout << (verify(h_C_cpu, h_C_gpu) ? "Correct" : "Wrong") << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
