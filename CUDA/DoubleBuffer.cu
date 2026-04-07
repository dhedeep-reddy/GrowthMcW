#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16

__global__ void matrixMulDoubleBuffered(int* A, int* B, int* C,
                                        int rowsA, int colsA, int colsB)
{
    __shared__ int tileA[2][TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    int numTiles = (colsA + TILE_SIZE - 1) / TILE_SIZE;

    int curr = 0, next = 1;

    // 🔥 Preload first tile
    int t = 0;
    if(row < rowsA && (t * TILE_SIZE + threadIdx.x) < colsA)
        tileA[curr][threadIdx.y][threadIdx.x] =
            A[row * colsA + t * TILE_SIZE + threadIdx.x];
    else
        tileA[curr][threadIdx.y][threadIdx.x] = 0;

    if(col < colsB && (t * TILE_SIZE + threadIdx.y) < colsA)
        tileB[curr][threadIdx.y][threadIdx.x] =
            B[(t * TILE_SIZE + threadIdx.y) * colsB + col];
    else
        tileB[curr][threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    for(t = 0; t < numTiles; t++)
    {
        // 🔥 Preload next tile (if exists)
        if(t + 1 < numTiles)
        {
            if(row < rowsA && ((t+1)*TILE_SIZE + threadIdx.x) < colsA)
                tileA[next][threadIdx.y][threadIdx.x] =
                    A[row * colsA + (t+1)*TILE_SIZE + threadIdx.x];
            else
                tileA[next][threadIdx.y][threadIdx.x] = 0;

            if(col < colsB && ((t+1)*TILE_SIZE + threadIdx.y) < colsA)
                tileB[next][threadIdx.y][threadIdx.x] =
                    B[((t+1)*TILE_SIZE + threadIdx.y) * colsB + col];
            else
                tileB[next][threadIdx.y][threadIdx.x] = 0;
        }

        // 🔥 Compute using current buffer
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[curr][threadIdx.y][k] *
                   tileB[curr][k][threadIdx.x];
        }

        __syncthreads();

        // 🔄 Swap buffers
        curr ^= 1;
        next ^= 1;
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

    matrixMulDoubleBuffered<<<blocks, threads>>>(d_A, d_B, d_C, N, N, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result Matrix:" << endl;

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
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