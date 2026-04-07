#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

__global__ void conv2D_naive(unsigned char* input,
                             unsigned char* output,
                             float* kernel,
                             int width, int height, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pad = K / 2;

    if (x < width && y < height)
    {
        float sum = 0.0f;

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < K; j++)
            {
                int nx = x + j - pad;
                int ny = y + i - pad;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    sum += input[ny * width + nx] * kernel[i * K + j];
                }
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
        cout << "Image not found!\n";
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    int width = gray.cols;
    int height = gray.rows;

    size_t size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    float *d_kernel;

    // Example kernel
    float h_kernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    cudaMemcpy(d_input, gray.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    conv2D_naive<<<grid, block>>>(d_input, d_output, d_kernel, width, height, 3);

    cudaMemcpy(gray.data, d_output, size, cudaMemcpyDeviceToHost);

    imwrite("output_cuda.jpg", gray);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    cout << "Done! Output saved as output_cuda.jpg\n";

    return 0;
}