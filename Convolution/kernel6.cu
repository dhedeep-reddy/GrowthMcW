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

__global__ void conv2D_double_buffer(unsigned char* __restrict__ input,
                                     unsigned char* __restrict__ output,
                                     int width, int height)
{

    __shared__ unsigned char tile[2][TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int curr = 0;
    int next = 1;

    auto loadStrip = [&](int buf, int strip_col, int strip_row) {
        for (int i = ty; i < TILE_SIZE + 2*RADIUS; i += blockDim.y) {
            for (int j = tx * 4; j < TILE_SIZE + 2*RADIUS; j += blockDim.x * 4) {
                int gx = strip_col + j - RADIUS;
                int gy = strip_row + i - RADIUS;

                int idx = gy * width + gx;

                if (gx >= 0 && gx + 3 < width && gy >= 0 && gy < height && (idx % 4 == 0)) {
                    uchar4 val = *((uchar4*)&input[idx]);
                    tile[buf][i][j]     = val.x;
                    tile[buf][i][j + 1] = val.y;
                    tile[buf][i][j + 2] = val.z;
                    tile[buf][i][j + 3] = val.w;
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int gxk = gx + k;
                        if (gxk >= 0 && gxk < width && gy >= 0 && gy < height)
                            tile[buf][i][j + k] = __ldg(&input[gy * width + gxk]);
                        else
                            tile[buf][i][j + k] = 0;
                    }
                }
            }
        }
    };

    int base_col = blockIdx.x * TILE_SIZE;   
    int base_row = blockIdx.y * TILE_SIZE;   

    loadStrip(curr, base_col, base_row);
    __syncthreads();

    int next_row = base_row + TILE_SIZE;     
    if (next_row < height) {
        loadStrip(next, base_col, next_row); 
    }

    int x_o = base_col + tx * WPT;
    int y_o = base_row + ty;

    int shared_x = tx * WPT + RADIUS;
    int shared_y = ty + RADIUS;

    float sum[WPT] = {0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < K; i++) {
        #pragma unroll
        for (int j = 0; j < K; j++) {
            float kval = d_kernel[i * K + j];
            #pragma unroll
            for (int w = 0; w < WPT; w++) {
                sum[w] += tile[curr][shared_y + i - RADIUS]
                                    [shared_x + w + j - RADIUS] * kval;
            }
        }
    }

    #pragma unroll
    for (int w = 0; w < WPT; w++) {
        if (x_o + w < width && y_o < height)
            output[y_o * width + x_o + w] = (unsigned char)min(max(sum[w], 0.0f), 255.0f);
    }

    __syncthreads();          
    curr ^= 1;
    next ^= 1;

    if (next_row < height) {

        x_o = base_col + tx * WPT;
        y_o = next_row + ty;
        shared_x = tx * WPT + RADIUS;
        shared_y = ty + RADIUS;

        float sum2[WPT] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < K; i++) {
            #pragma unroll
            for (int j = 0; j < K; j++) {
                float kval = d_kernel[i * K + j];
                #pragma unroll
                for (int w = 0; w < WPT; w++) {
                    sum2[w] += tile[curr][shared_y + i - RADIUS]
                                        [shared_x + w + j - RADIUS] * kval;
                }
            }
        }
        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            if (x_o + w < width && y_o < height)
                output[y_o * width + x_o + w] = (unsigned char)min(max(sum2[w], 0.0f), 255.0f);
        }
    }
}

int main()
{
    Mat img = imread("rose.jpg");
    if (img.empty()) {
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
    size_t imgSize = (size_t)width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input,  imgSize);
    cudaMalloc(&d_output, imgSize);

    cudaMemcpy(d_input, padded.data, imgSize, cudaMemcpyHostToDevice);

    float h_kernel[K*K] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, K*K*sizeof(float));

    dim3 block(TILE_SIZE / WPT, TILE_SIZE);
    dim3 grid((width  + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);

    conv2D_double_buffer<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(padded.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    Mat result = padded(Rect(RADIUS, RADIUS, gray.cols, gray.rows));
    imwrite("output_double_buffer.jpg", result);

    cout << "DOUBLE BUFFER VERSION RUN SUCCESSFULLY!\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}