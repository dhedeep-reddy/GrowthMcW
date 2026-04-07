#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// ================= CONFIG =================
#define TSM 64
#define TSN 64
#define TSK 16

#define WPTN 4
#define RTSN (TSN / WPTN)

// ==========================================

// Transpose kernel
__global__ void transposeKernel(float* input, float* output, int rows, int cols)
{
    __shared__ float tile[32][32];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads();

    int newX = blockIdx.y * 32 + threadIdx.x;
    int newY = blockIdx.x * 32 + threadIdx.y;

    if (newX < rows && newY < cols)
        output[newY * rows + newX] = tile[threadIdx.x][threadIdx.y];
}

// FINAL GEMM KERNEL
__global__ void gemmFinal(float* A, float* Bt, float* C,
                          int M, int N, int K)
{
    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSN][TSK + 2]; // padding

    int row = threadIdx.x;   // 0..TSM
    int col = threadIdx.y;   // 0..RTSN

    int globalRow = blockIdx.x * TSM + row;
    int globalCol = blockIdx.y * TSN + col;

    float acc[WPTN] = {0};

    for (int t = 0; t < K / TSK; t++)
    {
        int tiledIndex = t * TSK + col;

        // Load A
        if (globalRow < M && tiledIndex < K)
            Asub[col][row] = A[tiledIndex * M + globalRow];
        else
            Asub[col][row] = 0.0f;

        // Load Bᵀ
        if (globalCol < N && tiledIndex < K)
            Bsub[row][col] = Bt[tiledIndex * N + globalCol];
        else
            Bsub[row][col] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < TSK; k++)
        {
            float aVal = Asub[k][row];

            for (int w = 0; w < WPTN; w++)
            {
                acc[w] += aVal *
                          Bsub[col + w * RTSN][k];
            }
        }

        __syncthreads();
    }

    // Store
    for (int w = 0; w < WPTN; w++)
    {
        int cCol = globalCol + w * RTSN;

        if (globalRow < M && cCol < N)
            C[globalRow * N + cCol] = acc[w];
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

    // 🔹 Transpose B
    dim3 tBlock(32, 32);
    dim3 tGrid((N + 31)/32, (N + 31)/32);

    transposeKernel<<<tGrid, tBlock>>>(d_B, d_Bt, N, N);

    // 🔹 GEMM launch
    dim3 threads(TSM, RTSN);
    dim3 blocks(N / TSM, N / TSN);

    gemmFinal<<<blocks, threads>>>(d_A, d_Bt, d_C, N, N, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result Matrix (5x5):\n";

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            cout << h_C[i * N + j] << " ";
        }
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