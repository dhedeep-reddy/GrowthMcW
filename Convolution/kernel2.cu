#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define TILE_SIZE 16
#define K 3
#define RADIUS (K/2)
#define WPT 2   

__constant__ float d_kernel[K*K];

__global__ void conv2D(unsigned char* input, unsigned char* output,
                       int width, int height)
{
    __shared__ unsigned char tile[TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_x = blockIdx.x * TILE_SIZE + tx * WPT;
    int y = blockIdx.y * TILE_SIZE + ty;

    int shared_y = ty + RADIUS;

    for(int i = tx; i < TILE_SIZE + 2*RADIUS; i += blockDim.x)
    {
        int global_x = blockIdx.x * TILE_SIZE + i - RADIUS;

        if (global_x >= 0 && global_x < width && y < height)
            tile[shared_y][i] = input[y * width + global_x];
        else
            tile[shared_y][i] = 0;
    }

    if (ty < RADIUS)
    {
        int global_y_top = y - RADIUS;
        int global_y_bottom = y + TILE_SIZE;

        for(int i = tx; i < TILE_SIZE + 2*RADIUS; i += blockDim.x)
        {
            int global_x = blockIdx.x * TILE_SIZE + i - RADIUS;

            // Top halo
            if (global_y_top >= 0 && global_x >= 0 && global_x < width)
                tile[ty][i] = input[global_y_top * width + global_x];
            else
                tile[ty][i] = 0;

            // Bottom halo
            if (global_y_bottom < height && global_x >= 0 && global_x < width)
                tile[ty + TILE_SIZE + RADIUS][i] =
                    input[global_y_bottom * width + global_x];
            else
                tile[ty + TILE_SIZE + RADIUS][i] = 0;
        }
    }

    __syncthreads();

    float sum[WPT] = {0};

    for(int w = 0; w < WPT; w++)
    {
        int col = base_x + w;

        if (col < width && y < height)
        {
            for(int ky = 0; ky < K; ky++)
            {
                for(int kx = 0; kx < K; kx++)
                {
                    sum[w] += tile[shared_y + ky - RADIUS]
                                   [tx * WPT + w + kx] *
                              d_kernel[ky * K + kx];
                }
            }
        }
    }

    for(int w = 0; w < WPT; w++)
    {
        int col = base_x + w;

        if (col < width && y < height)
        {
            output[y * width + col] = min(max(int(sum[w]), 0), 255);
        }
    }
}

int main()
{
    Mat img = imread("rose.jpg", IMREAD_GRAYSCALE);
    if(img.empty())
    {
        cout << "Image not found\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    cudaMemcpy(d_input, img.data, width * height, cudaMemcpyHostToDevice);

    float h_kernel[K*K] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    cudaMemcpyToSymbol(d_kernel, h_kernel, K*K*sizeof(float));

    dim3 block(TILE_SIZE / WPT, TILE_SIZE);  // (8,16)
    dim3 grid((width + TILE_SIZE - 1)/TILE_SIZE,
              (height + TILE_SIZE - 1)/TILE_SIZE);

    conv2D<<<grid, block>>>(d_input, d_output, width, height);

    Mat output(height, width, CV_8UC1);
    cudaMemcpy(output.data, d_output, width * height, cudaMemcpyDeviceToHost);

    imwrite("output_register.jpg", output);

    cudaFree(d_input);
    cudaFree(d_output);

    cout << "Done. Output saved as output_fixed.jpg\n";
    return 0;
}   