#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;
using namespace cv;

__device__ float cubic(float x)
{
    float a = -0.75f;
    x = fabs(x);

    if (x <= 1.0f)
        return (a + 2)*x*x*x - (a + 3)*x*x + 1;
    else if (x < 2.0f)
        return a*x*x*x - 5*a*x*x + 8*a*x - 4*a;
    else
        return 0.0f;
}

__global__ void bicubic_kernel(float* input, float* output,
                               int inW, int inH, int outW, int outH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    float in_x = (x + 0.5f) * scaleX - 0.5f;
    float in_y = (y + 0.5f) * scaleY - 0.5f;

    in_x = min(max(in_x, 0.0f), inW - 1.0f);
    in_y = min(max(in_y, 0.0f), inH - 1.0f);

    int ix = floor(in_x);
    int iy = floor(in_y);

    float dx = in_x - ix;
    float dy = in_y - iy;

    float Wx[4], Wy[4];

    for(int i = -1; i <= 2; i++)
        Wx[i+1] = cubic(i - dx);

    for(int i = -1; i <= 2; i++)
        Wy[i+1] = cubic(dy - i);

    float sum = 0.0f;

    for(int m = -1; m <= 2; m++)
    {
        for(int n = -1; n <= 2; n++)
        {
            int px = min(max(ix + n, 0), inW - 1);
            int py = min(max(iy + m, 0), inH - 1);

            float pixel = input[py * inW + px];
            sum += pixel * Wx[n+1] * Wy[m+1];
        }
    }

    output[y * outW + x] = sum;
}

void validate(Mat& A, Mat& B)
{
    double error = 0, max_error = 0;

    for(int y = 0; y < A.rows; y++)
    {
        for(int x = 0; x < A.cols; x++)
        {
            float diff = abs(A.at<float>(y,x) - B.at<float>(y,x));
            error += diff;
            max_error = max(max_error, (double)diff);
        }
    }

    cout << "Total Error: " << error << endl;
    cout << "Max Error: " << max_error << endl;
    cout << "Mean Error: " << error / (A.rows * A.cols) << endl;
}

int main()
{
    Mat img = imread("rose.jpg");
    if(img.empty())
    {
        cout << "Image not found\n";
        return -1;
    }

    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int inW = gray.cols;
    int inH = gray.rows;
    int outW = inW * 2;
    int outH = inH * 2;

    
    Mat opencv_out;
    resize(gray, opencv_out, Size(outW, outH), 0, 0, INTER_CUBIC);

    
    float *d_input, *d_output;
    cudaMalloc(&d_input, inW * inH * sizeof(float));
    cudaMalloc(&d_output, outW * outH * sizeof(float));

    cudaMemcpy(d_input, gray.ptr<float>(), inW * inH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((outW+15)/16, (outH+15)/16);

    bicubic_kernel<<<grid, block>>>(d_input, d_output, inW, inH, outW, outH);
    cudaDeviceSynchronize();

    Mat cuda_out(outH, outW, CV_32F);
    cudaMemcpy(cuda_out.ptr<float>(), d_output, outW * outH * sizeof(float), cudaMemcpyDeviceToHost);

  
    validate(opencv_out, cuda_out);

   
    opencv_out.convertTo(opencv_out, CV_8U);
    cuda_out.convertTo(cuda_out, CV_8U);

    imwrite("opencv.jpg", opencv_out);
    imwrite("cuda.jpg", cuda_out);

    cudaFree(d_input);
    cudaFree(d_output);

    cout << "Images saved: opencv.jpg, cuda.jpg\n";

    return 0;
}