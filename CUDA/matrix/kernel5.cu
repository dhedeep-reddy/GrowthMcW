#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

#define N 2000   

#define TILE_M  32
#define TILE_N  32
#define TILE_K  64
#define WPT      4

#define RTS_N  (TILE_N / WPT)

static_assert(TILE_N % WPT == 0, "TILE_N must be divisible by WPT");
static_assert(TILE_K % WPT == 0, "TILE_K must be divisible by WPT");

inline int pad_to(int x, int mult) { return ((x + mult - 1) / mult) * mult; }

#define PAD_UNIT TILE_K


#define CFLOAT4(ptr) (reinterpret_cast<const float4*>(ptr))

__global__ void gpu_matmul_optimized(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float*       __restrict__ C,
                                     int N,      
                                     int Np)    
{
    __shared__ float As[2][TILE_M][TILE_K];
    __shared__ float Bs[2][TILE_K][TILE_N + 2];  

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row     = blockIdx.y * TILE_M + ty;
    const int colBase = blockIdx.x * TILE_N + tx * WPT;

    float sum[WPT] = {0.0f};

    // numTiles is exact because Np is padded to a multiple of TILE_K
    const int numTiles = Np / TILE_K;
    int ping = 0;


    auto load_tile = [&](int buf, int t) {
        const int tid      = ty * RTS_N + tx;
        const int nThreads = TILE_M * RTS_N;

        const int nVecA = (TILE_M * TILE_K) / WPT;
        for (int i = tid; i < nVecA; i += nThreads) {
            int lrow = i / (TILE_K / WPT);
            int lcol = i % (TILE_K / WPT);
            int grow = blockIdx.y * TILE_M + lrow;   
            int gk4  = t * (TILE_K / WPT) + lcol;   

            float4 a4 = CFLOAT4(A)[grow * (Np / WPT) + gk4];
            As[buf][lrow][lcol*WPT+0] = a4.x;
            As[buf][lrow][lcol*WPT+1] = a4.y;
            As[buf][lrow][lcol*WPT+2] = a4.z;
            As[buf][lrow][lcol*WPT+3] = a4.w;
        }

        const int nVecB = (TILE_K * TILE_N) / WPT;
        for (int i = tid; i < nVecB; i += nThreads) {
            int lk    = i / (TILE_N / WPT);
            int lcol  = i % (TILE_N / WPT);
            int gk    = t * TILE_K + lk;             
            int gcol4 = blockIdx.x * (TILE_N / WPT) + lcol;

            float4 b4 = CFLOAT4(B)[gk * (Np / WPT) + gcol4];
            Bs[buf][lk][lcol*WPT+0] = b4.x;
            Bs[buf][lk][lcol*WPT+1] = b4.y;
            Bs[buf][lk][lcol*WPT+2] = b4.z;
            Bs[buf][lk][lcol*WPT+3] = b4.w;
        }
    };

    load_tile(ping, 0);
    __syncthreads();

    for (int t = 0; t < numTiles; t++) {
        const int pong = 1 - ping;

        // Prefetch next tile into pong buffer (overlap with compute on ping)
        if (t + 1 < numTiles)
            load_tile(pong, t + 1);

        // Compute on ping buffer
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[ping][ty][k];
         
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

void cpu_matmul(const float* A, const float* B, float* C, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int k = 0; k < n; k++)
                s += A[i*n + k] * B[k*n + j];
            C[i*n + j] = s;
        }
}

bool verify(const float* cpu, const float* gpu, int n)
{
    for (int i = 0; i < n*n; i++)
        if (fabs(cpu[i] - gpu[i]) > 1e-2f) {
            printf("Mismatch at [%d,%d]: cpu=%.4f gpu=%.4f\n",
                   i/n, i%n, cpu[i], gpu[i]);
            return false;
        }
    return true;
}

int main()
{
    const int Np = pad_to(N, PAD_UNIT);  

    printf("N=%d  Np=%d (padded to multiple of %d)\n", N, Np, PAD_UNIT);

    size_t size_real = (size_t)N  * N  * sizeof(float);
    size_t size_pad  = (size_t)Np * Np * sizeof(float);

    float *h_A     = (float*)malloc(size_real);
    float *h_B     = (float*)malloc(size_real);
    float *h_C_cpu = (float*)malloc(size_real);
    float *h_C_gpu = (float*)malloc(size_real);

    srand(42);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = rand() % 5;
        h_B[i] = rand() % 5;
    }

    cpu_matmul(h_A, h_B, h_C_cpu, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_pad);
    cudaMalloc(&d_B, size_pad);
    cudaMalloc(&d_C, size_pad);
    cudaMemset(d_A, 0, size_pad);
    cudaMemset(d_B, 0, size_pad);
    cudaMemset(d_C, 0, size_pad);

    
    for (int i = 0; i < N; i++) {
        cudaMemcpy(d_A + (size_t)i * Np, h_A + (size_t)i * N,
                   N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B + (size_t)i * Np, h_B + (size_t)i * N,
                   N * sizeof(float), cudaMemcpyHostToDevice);
    }

    dim3 threads(RTS_N, TILE_M);
    dim3 blocks(Np / TILE_N, Np / TILE_M);

    printf("Grid  : (%d, %d)\n", blocks.x, blocks.y);
    printf("Block : (%d, %d) = %d threads\n", threads.x, threads.y,
           threads.x * threads.y);

    size_t smem = 2 * (TILE_M*TILE_K + TILE_K*(TILE_N+2)) * sizeof(float);
    printf("Shared mem/block: %.2f KB\n", smem / 1024.0f);

    gpu_matmul_optimized<<<blocks, threads>>>(d_A, d_B, d_C, N, Np);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        
    cudaMemcpy(h_C_gpu, d_C, size_real, cudaMemcpyDeviceToHost);

    cout << (verify(h_C_cpu, h_C_gpu, N) ? "Correct" : "Wrong") << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
