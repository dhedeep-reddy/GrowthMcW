// remove padding condition

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define TILE_SIZE 16
#define K 3
#define RADIUS (K/2)

__constant__ float d_kernel[K*K];

__global__ void conv2D_padded(unsigned char* input,
                              unsigned char* output,
                              int width, int height)
{
    __shared__ unsigned char tile[TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_x = blockIdx.x * TILE_SIZE;
    int base_y = blockIdx.y * TILE_SIZE;

    int x = base_x + tx * 2;
    int y = base_y + ty;

    for(int i = ty; i < TILE_SIZE + 2*RADIUS; i += blockDim.y)
    {
        for(int j = tx; j < TILE_SIZE + 2*RADIUS; j += blockDim.x)
        {
            int global_x = base_x + j - RADIUS;
            int global_y = base_y + i - RADIUS;

            tile[i][j] = input[global_y * width + global_x];
        }
    }

    __syncthreads();

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    #pragma unroll
    for (int i = 0; i < K; i++)
    {
        #pragma unroll
        for (int j = 0; j < K; j++)
        {
            float k = d_kernel[i*K + j];

            sum1 += tile[ty + i][tx*2 + j] * k;
            sum2 += tile[ty + i][tx*2 + j + 1] * k;
        }
    }

    output[y * width + x]     = min(max(sum1, 0.0f), 255.0f);
    output[y * width + x + 1] = min(max(sum2, 0.0f), 255.0f);
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

    dim3 block(TILE_SIZE/2, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1)/TILE_SIZE,
              (height + TILE_SIZE - 1)/TILE_SIZE);

    conv2D_padded<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(padded.data, d_output, size, cudaMemcpyDeviceToHost);

    Mat result = padded(Rect(RADIUS, RADIUS,
                             gray.cols, gray.rows));

    imwrite("output_padded.jpg", result);

    cout << "Done! Output saved as output_padded.jpg\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}