#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16

__global__ void matrixMulVectorized(int* A, int* B, int* C,
                                    int rowsA, int colsA, int colsB)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    for(int t = 0; t < (colsA + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        int tiledColA = t * TILE_SIZE + threadIdx.x;
        int tiledRowB = t * TILE_SIZE + threadIdx.y;

        // ===================== LOAD A =====================
        int indexA = row * colsA + tiledColA;

        if(row < rowsA && tiledColA < colsA)
        {
            if((indexA % 4 == 0) && threadIdx.x <= TILE_SIZE - 4 && tiledColA + 3 < colsA)
            {
                int4 val = *(int4*)&A[indexA];

                tileA[threadIdx.y][threadIdx.x]     = val.x;
                tileA[threadIdx.y][threadIdx.x + 1] = val.y;
                tileA[threadIdx.y][threadIdx.x + 2] = val.z;
                tileA[threadIdx.y][threadIdx.x + 3] = val.w;
            }
            else
            {
                tileA[threadIdx.y][threadIdx.x] = A[indexA];
            }
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        // ===================== LOAD B =====================
        int indexB = tiledRowB * colsB + col;

        if(col < colsB && tiledRowB < colsA)
        {
            if((indexB % 4 == 0) && threadIdx.y <= TILE_SIZE - 4 && tiledRowB + 3 < colsA)
            {
                int4 val = *(int4*)&B[indexB];

                tileB[threadIdx.y][threadIdx.x]     = val.x;
                tileB[threadIdx.y + 1][threadIdx.x] = val.y;
                tileB[threadIdx.y + 2][threadIdx.x] = val.z;
                tileB[threadIdx.y + 3][threadIdx.x] = val.w;
            }
            else
            {
                tileB[threadIdx.y][threadIdx.x] = B[indexB];
            }
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // ===================== COMPUTE =====================
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < rowsA && col < colsB)
        C[row * colsB + col] = sum;
}

int main()
{
    const int N = 1024;

    int size = N * N * sizeof(int);

    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];

    for(int i = 0; i < N * N; i++)
    {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMulVectorized<<<blocks, threads>>>(d_A, d_B, d_C, N, N, N);

    cudaDeviceSynchronize(); // important for catching errors

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result (first 5x5):" << endl;
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            cout << h_C[i * N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}