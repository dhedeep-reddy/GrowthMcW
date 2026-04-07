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

__global__ void conv2D_optimized(unsigned char* input,
                                 unsigned char* output,
                                 int width, int height)
{
    __shared__ unsigned char tile[TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x_o = blockIdx.x * TILE_SIZE + tx * WPT;
    int y_o = blockIdx.y * TILE_SIZE + ty;

    int shared_x = tx * WPT + RADIUS;
    int shared_y = ty + RADIUS;

    #pragma unroll
    for (int i = ty; i < TILE_SIZE + 2*RADIUS; i += blockDim.y)
    {
        #pragma unroll
        for (int j = tx * 4; j < TILE_SIZE + 2*RADIUS; j += blockDim.x * 4)
        {
            int gx = blockIdx.x * TILE_SIZE + j - RADIUS;
            int gy = blockIdx.y * TILE_SIZE + i - RADIUS;
            int idx = gy * width + gx;

            if (gx >= 0 && gx + 3 < width && gy >= 0 && gy < height && (idx % 4 == 0))
            {
                uchar4 val = *((uchar4*)&input[idx]);
                tile[i][j]     = val.x;
                tile[i][j + 1] = val.y;
                tile[i][j + 2] = val.z;
                tile[i][j + 3] = val.w;
            }
            else
            {
                for(int k = 0; k < 4; k++)
                {
                    if (gx + k < width && gy < height && gx + k >= 0 && gy >= 0)
                        tile[i][j + k] = __ldg(&input[gy * width + gx + k]);
                    else
                        tile[i][j + k] = 0;
                }
            }
        }
    }

    __syncthreads();

    float sum[WPT] = {0};

    #pragma unroll
    for (int i = 0; i < K; i++)
    {
        #pragma unroll
        for (int j = 0; j < K; j++)
        {
            float k = d_kernel[i*K + j];

            #pragma unroll
            for (int w = 0; w < WPT; w++)
            {
                sum[w] += tile[shared_y + i - RADIUS][shared_x + w + j - RADIUS] * k;
            }
        }
    }

    #pragma unroll
    for (int w = 0; w < WPT; w++)
    {
        if (x_o + w < width && y_o < height)
        {
            output[y_o * width + x_o + w] =
                min(max(sum[w], 0.0f), 255.0f);
        }
    }
}

// Main
int main()
{
    Mat img = imread("rose.jpg");
    if (img.empty())
    {
        cout << "Error loading image\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat padded;
    copyMakeBorder(gray, padded,
                   RADIUS, RADIUS, RADIUS, RADIUS,
                   BORDER_CONSTANT, 0);

    int width  = padded.cols;
    int height = padded.rows;
    size_t size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, padded.data, size, cudaMemcpyHostToDevice);

    float h_kernel[K*K] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    cudaMemcpyToSymbol(d_kernel, h_kernel, K*K*sizeof(float));

    dim3 block(TILE_SIZE / WPT, TILE_SIZE);  
    dim3 grid((width + TILE_SIZE - 1)/TILE_SIZE,
              (height + TILE_SIZE - 1)/TILE_SIZE);

    conv2D_optimized<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(padded.data, d_output, size, cudaMemcpyDeviceToHost);

    Mat result = padded(Rect(RADIUS, RADIUS,
                             gray.cols, gray.rows));

    imwrite("output_optimized.jpg", result);

    cout << "OPTIMIZED VERSION RUN SUCCESSFULLY!\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}