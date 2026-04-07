#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define TILE_SIZE 64
#define K 3
#define RADIUS (K/2)

#define WPT 8   

__constant__ float d_kernel[K*K];

__global__ void conv2D_double_buffer(unsigned char* input,
                                     unsigned char* output,
                                     int width, int height)
{
    __shared__ unsigned char tile[2][TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_x = blockIdx.x * TILE_SIZE;
    int base_y = blockIdx.y * TILE_SIZE;

    int x = base_x + tx * WPT;
    int y = base_y + ty;

    int curr = 0;
    int next = 1;  

    for(int i = ty; i < TILE_SIZE + 2*RADIUS; i += blockDim.y)
    {
        for(int j = tx; j < TILE_SIZE + 2*RADIUS; j += blockDim.x)
        {
            int gx = base_x + j - RADIUS;
            int gy = base_y + i - RADIUS;

            if(gx >= 0 && gx < width && gy >= 0 && gy < height)
                tile[curr][i][j] = input[gy * width + gx];
            else
                tile[curr][i][j] = 0;
        }
    }

    __syncthreads();

    int next_base_y = base_y + TILE_SIZE;
    for(int i = ty; i < TILE_SIZE + 2*RADIUS; i += blockDim.y)
    {
        for(int j = tx; j < TILE_SIZE + 2*RADIUS; j += blockDim.x)
        {
            int gx = base_x + j - RADIUS;
            int gy = next_base_y + i - RADIUS;

            if(gx >= 0 && gx < width && gy >= 0 && gy < height)
                tile[next][i][j] = input[gy * width + gx];
            else
                tile[next][i][j] = 0;
        }
    }

    float sum[WPT] = {0};

    #pragma unroll
    for (int i = 0; i < K; i++)
    {
        #pragma unroll
        for (int j = 0; j < K; j++)
        {
            float k = d_kernel[i*K + j];

            #pragma unroll
            for(int w = 0; w < WPT; w++)
            {
                sum[w] += tile[curr][ty + i][tx*WPT + j + w] * k;
            }
        }
    }

    #pragma unroll
    for(int w = 0; w < WPT; w++)
    {
        if(x + w < width && y < height)
        {
            output[y * width + x + w] =
                min(max(sum[w], 0.0f), 255.0f);
        }
    }

    __syncthreads();
    curr ^= 1;
    next ^= 1;

}
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

    conv2D_double_buffer<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(padded.data, d_output, size, cudaMemcpyDeviceToHost);

    Mat result = padded(Rect(RADIUS, RADIUS,
                             gray.cols, gray.rows));

    imwrite("output_double_buffer_WPT4.jpg", result);

    cout << "Done! Output saved as output_double_buffer.jpg\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}