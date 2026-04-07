#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define TILE_SIZE 16
#define K 3
#define RADIUS (K/2)

__constant__ float d_kernel[K*K];

__global__ void conv2D_shared(unsigned char* input,
                              unsigned char* output,
                              int width, int height)
{
    __shared__ unsigned char tile[TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int tile_start_x = blockIdx.x * TILE_SIZE - RADIUS;
    int tile_start_y = blockIdx.y * TILE_SIZE - RADIUS;

    for(int i = ty; i < TILE_SIZE + 2*RADIUS; i += blockDim.y)
    {
        for(int j = tx; j < TILE_SIZE + 2*RADIUS; j += blockDim.x)
        {
            int global_x = tile_start_x + j;
            int global_y = tile_start_y + i;

            if(global_x >= 0 && global_x < width &&
               global_y >= 0 && global_y < height)
            {
                tile[i][j] = input[global_y * width + global_x];
            }
            else
            {
                tile[i][j] = 0;
            }
        }
    }

    __syncthreads();

    if (x < width && y < height)
    {
        float sum = 0.0f;

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                sum += tile[ty + i][tx + j] * d_kernel[i*K + j];
            }
        }

        output[y * width + x] = min(max(sum, 0.0f), 255.0f);
    }
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

    int width = gray.cols;
    int height = gray.rows;

    size_t size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, gray.data, size, cudaMemcpyHostToDevice);

    float h_kernel[K*K] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    cudaMemcpyToSymbol(d_kernel, h_kernel, K*K*sizeof(float));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1)/TILE_SIZE,
              (height + TILE_SIZE - 1)/TILE_SIZE);

    conv2D_shared<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(gray.data, d_output, size, cudaMemcpyDeviceToHost);

    imwrite("output_shared.jpg", gray);

    cout << "Done! Output saved as output_shared_fixed.jpg\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}