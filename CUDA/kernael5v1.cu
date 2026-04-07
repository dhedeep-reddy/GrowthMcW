#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// ================= CONFIG =================
#define TSM 16
#define TSN 16
#define TSK 16

#define WPTN 4
#define RTSN (TSN / WPTN)   // 16/4 = 4
#define LPT (TSK / RTSN)    // 16/4 = 4
// ==========================================


// ---------------- TRANSPOSE ----------------
__global__ void transposeKernel(float* input, float* output, int rows, int cols)
{
    __shared__ float tile[TSM][TSN];

    int x = blockIdx.x * TSN + threadIdx.x;
    int y = blockIdx.y * TSM + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads();

    int newX = blockIdx.y * TSM + threadIdx.x;
    int newY = blockIdx.x * TSN + threadIdx.y;

    if (newX < rows && newY < cols)
        output[newY * rows + newX] = tile[threadIdx.x][threadIdx.y];
}


// ---------------- GEMM ----------------
__global__ void gemm16(float* A, float* Bt, float* C,
                       int M, int N, int K)
{
    __shared__ float Asub[TSK][TSM];
    __shared__ float Bsub[TSN][TSK + 1]; // padding to avoid bank conflicts

    int row = threadIdx.x;   // 0..15
    int col = threadIdx.y;   // 0..3

    int globalRow = blockIdx.x * TSM + row;
    int globalCol = blockIdx.y * TSN + col;

    float acc[WPTN] = {0};

    for (int t = 0; t < K / TSK; t++)
    {
        // ✅ LOADS PER THREAD (FIXED)
        for (int l = 0; l < LPT; l++)
        {
            int tiledCol = t * TSK + col + l * RTSN;

            // Load A
            if (globalRow < M && tiledCol < K)
                Asub[col + l * RTSN][row] = A[globalRow * K + tiledCol];
            else
                Asub[col + l * RTSN][row] = 0.0f;

            // Load Bᵀ
            if (globalCol < N && tiledCol < K)
                Bsub[row][col + l * RTSN] = Bt[globalCol * K + tiledCol];
            else
                Bsub[row][col + l * RTSN] = 0.0f;
        }

        __syncthreads();

        // ✅ COMPUTE
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

    // ✅ STORE RESULTS
    for (int w = 0; w < WPTN; w++)
    {
        int cCol = globalCol + w * RTSN;

        if (globalRow < M && cCol < N)
            C[globalRow * N + cCol] = acc[w];
    }
}


// ---------------- MAIN ----------------
int main()
{
    const int N = 1024;

    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize
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
    dim3 tBlock(TSN, TSM);  // (16,16)
    dim3 tGrid((N + TSN - 1)/TSN, (N + TSM - 1)/TSM);

    transposeKernel<<<tGrid, tBlock>>>(d_B, d_Bt, N, N);

    // 🔹 GEMM
    dim3 threads(TSM, RTSN);   // (16,4)
    dim3 blocks(N / TSM, N / TSN);

    gemm16<<<blocks, threads>>>(d_A, d_Bt, d_C, N, N, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print small portion
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