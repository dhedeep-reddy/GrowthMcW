#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// ================= CONFIG =================
#define TSM 64
#define TSN 64
#define TSK 16

#define WPTM 4
#define WPTN 4

#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)

#define WIDTH 4   // float4
// ==========================================

// TRANSPOSE (for coalescing)
__global__ void transpose(float* A, float* At, int N)
{
    __shared__ float tile[32][32];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];

    __syncthreads();

    int newX = blockIdx.y * 32 + threadIdx.x;
    int newY = blockIdx.x * 32 + threadIdx.y;

    if (newX < N && newY < N)
        At[newY * N + newX] = tile[threadIdx.x][threadIdx.y];
}

// ================= GEMM =================
__global__ void gemm2D_vec(float* A, float* Bt, float* C,
                          int M, int N, int K)
{
    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSK][TSN];

    int tidm = threadIdx.x;
    int tidn = threadIdx.y;

    int globalRowBase = blockIdx.x * TSM;
    int globalColBase = blockIdx.y * TSN;

    float acc[WPTM][WPTN] = {0};

    for (int t = 0; t < K / TSK; t++)
    {
        // ================= LOAD (VECTORIZED) =================

        int linearTid = tidn * RTSM + tidm;

        for (int i = linearTid; i < (TSK * TSM) / WIDTH; i += RTSM * RTSN)
        {
            int row = i % (TSM / WIDTH);
            int col = i / (TSM / WIDTH);

            int tiledIndex = t * TSK + col;

            float4 vecA = ((float4*)A)[(tiledIndex * M + globalRowBase) / WIDTH + row];

            // Store into shared memory
            Asub[col][row * WIDTH + 0] = vecA.x;
            Asub[col][row * WIDTH + 1] = vecA.y;
            Asub[col][row * WIDTH + 2] = vecA.z;
            Asub[col][row * WIDTH + 3] = vecA.w;
        }

        for (int i = linearTid; i < (TSK * TSN) / WIDTH; i += RTSM * RTSN)
        {
            int row = i % (TSN / WIDTH);
            int col = i / (TSN / WIDTH);

            int tiledIndex = t * TSK + col;

            float4 vecB = ((float4*)Bt)[(tiledIndex * N + globalColBase) / WIDTH + row];

            Bsub[col][row * WIDTH + 0] = vecB.x;
            Bsub[col][row * WIDTH + 1] = vecB.y;
            Bsub[col][row * WIDTH + 2] = vecB.z;
            Bsub[col][row * WIDTH + 3] = vecB.w;
        }

        __syncthreads();

        // ================= COMPUTE =================
        for (int k = 0; k < TSK; k++)
        {
            float Areg[WPTM];
            float Breg[WPTN];

            for (int wm = 0; wm < WPTM; wm++)
            {
                int row = tidm + wm * RTSM;
                Areg[wm] = Asub[k][row];
            }

            for (int wn = 0; wn < WPTN; wn++)
            {
                int col = tidn + wn * RTSN;
                Breg[wn] = Bsub[k][col];
            }

            for (int wm = 0; wm < WPTM; wm++)
            {
                for (int wn = 0; wn < WPTN; wn++)
                {
                    acc[wm][wn] += Areg[wm] * Breg[wn];
                }
            }
        }

        __syncthreads();
    }

    // ================= STORE =================
    for (int wm = 0; wm < WPTM; wm++)
    {
        int globalRow = globalRowBase + tidm + wm * RTSM;

        for (int wn = 0; wn < WPTN; wn++)
        {
            int globalCol = globalColBase + tidn + wn * RTSN;

            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = acc[wm][wn];
        }
    }
}

// ================= MAIN =================
int main()
{
    const int N = 1024;
    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_Bt, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_Bt, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Transpose
    dim3 tBlock(32, 32);
    dim3 tGrid(N / 32, N / 32);
    transpose<<<tGrid, tBlock>>>(d_B, d_Bt, N);

    // GEMM
    dim3 threads(RTSM, RTSN);   // 16x16
    dim3 blocks(26, 26);        // SM aligned

    gemm2D_vec<<<blocks, threads>>>(d_A, d_Bt, d_C, N, N, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result (5x5):\n";
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            cout << h_C[i * N + j] << " ";
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Bt);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}