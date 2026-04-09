#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
using namespace std;
using namespace cv;


float cubic_cpu(float x)
{
    const float a = -0.75f;
    x = fabsf(x);
    if (x <= 1.0f)
        return (a + 2.f)*x*x*x - (a + 3.f)*x*x + 1.f;
    else if (x < 2.0f)
        return a*x*x*x - 5.f*a*x*x + 8.f*a*x - 4.f*a;
    else
        return 0.0f;
}

Mat bicubic_forward_cpu(const Mat& input, int outW, int outH)
{
    int inW = input.cols, inH = input.rows;
    Mat output(outH, outW, CV_32F);
    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    for (int y = 0; y < outH; y++)
    for (int x = 0; x < outW; x++)
    {
        float in_x = (x + 0.5f)*scaleX - 0.5f;
        float in_y = (y + 0.5f)*scaleY - 0.5f;
        in_x = max(0.f, min(in_x, (float)(inW-1)));
        in_y = max(0.f, min(in_y, (float)(inH-1)));

        int ix = (int)floorf(in_x), iy = (int)floorf(in_y);
        float dx = in_x - ix, dy = in_y - iy;

        float Wx[4], Wy[4];
        for (int i = -1; i <= 2; i++) Wx[i+1] = cubic_cpu(i - dx);
        for (int i = -1; i <= 2; i++) Wy[i+1] = cubic_cpu(dy - i);

        float sum = 0.f;
        for (int m = -1; m <= 2; m++)
        for (int n = -1; n <= 2; n++)
        {
            int px = min(max(ix+n, 0), inW-1);
            int py = min(max(iy+m, 0), inH-1);
            sum += input.at<float>(py,px) * Wx[n+1] * Wy[m+1];
        }
        output.at<float>(y,x) = sum;
    }
    return output;
}

Mat bicubic_backward_cpu(const Mat& input, const Mat& grad_output,
                         int outW, int outH)
{
    int inW = input.cols, inH = input.rows;
    Mat grad_input = Mat::zeros(inH, inW, CV_32F);
    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    for (int y = 0; y < outH; y++)
    for (int x = 0; x < outW; x++)
    {
        float in_x = (x + 0.5f)*scaleX - 0.5f;
        float in_y = (y + 0.5f)*scaleY - 0.5f;
        in_x = max(0.f, min(in_x, (float)(inW-1)));
        in_y = max(0.f, min(in_y, (float)(inH-1)));

        int ix = (int)floorf(in_x), iy = (int)floorf(in_y);
        float dx = in_x - ix, dy = in_y - iy;

        float Wx[4], Wy[4];
        for (int i = -1; i <= 2; i++) Wx[i+1] = cubic_cpu(i - dx);
        for (int i = -1; i <= 2; i++) Wy[i+1] = cubic_cpu(dy - i);

        float g = grad_output.at<float>(y,x);
        for (int m = -1; m <= 2; m++)
        for (int n = -1; n <= 2; n++)
        {
            int px = min(max(ix+n, 0), inW-1);
            int py = min(max(iy+m, 0), inH-1);
            grad_input.at<float>(py,px) += g * Wx[n+1] * Wy[m+1];
        }
    }
    return grad_input;
}


#define TILE 16

__device__ float cubic_gpu(float x)
{
    const float a = -0.75f;
    x = fabsf(x);
    if (x <= 1.0f)
        return (a + 2.f)*x*x*x - (a + 3.f)*x*x + 1.f;
    else if (x < 2.0f)
        return a*x*x*x - 5.f*a*x*x + 8.f*a*x - 4.f*a;
    else
        return 0.0f;
}

__global__ void bicubic_forward_kernel(
    const float* input,  int inW,  int inH,
          float* output, int outW, int outH,
    float scaleX, float scaleY)
{
    int out_x = blockIdx.x * TILE + threadIdx.x;
    int out_y = blockIdx.y * TILE + threadIdx.y;
    if (out_x >= outW || out_y >= outH) return;

    float in_x = (out_x + 0.5f) * scaleX - 0.5f;
    float in_y = (out_y + 0.5f) * scaleY - 0.5f;
    in_x = fmaxf(0.f, fminf(in_x, (float)(inW - 1)));
    in_y = fmaxf(0.f, fminf(in_y, (float)(inH - 1)));

    int ix = (int)floorf(in_x);
    int iy = (int)floorf(in_y);
    float dx = in_x - ix;
    float dy = in_y - iy;

    // compute 4 weights per direction individually
    float Wx[4], Wy[4];
    for (int i = 0; i < 4; i++) Wx[i] = cubic_gpu(i - 1 - dx);
    for (int i = 0; i < 4; i++) Wy[i] = cubic_gpu(i - 1 - dy);

    // 4x4 accumulation, each read hits global memory
    float sum = 0.f;
    for (int m = 0; m < 4; m++)
    for (int n = 0; n < 4; n++)
    {
        int gx = min(max(ix + n - 1, 0), inW - 1);
        int gy = min(max(iy + m - 1, 0), inH - 1);
        sum += input[gy * inW + gx] * Wx[n] * Wy[m];
    }
    output[out_y * outW + out_x] = sum;
}

__global__ void bicubic_backward_kernel(
    const float* grad_output, int outW, int outH,
          float* grad_input,  int inW,  int inH,
    float scaleX, float scaleY)
{
    int out_x = blockIdx.x * TILE + threadIdx.x;
    int out_y = blockIdx.y * TILE + threadIdx.y;
    if (out_x >= outW || out_y >= outH) return;

    float in_x = (out_x + 0.5f) * scaleX - 0.5f;
    float in_y = (out_y + 0.5f) * scaleY - 0.5f;
    in_x = fmaxf(0.f, fminf(in_x, (float)(inW - 1)));
    in_y = fmaxf(0.f, fminf(in_y, (float)(inH - 1)));

    int ix = (int)floorf(in_x);
    int iy = (int)floorf(in_y);
    float dx = in_x - ix;
    float dy = in_y - iy;

    float Wx[4], Wy[4];
    for (int i = 0; i < 4; i++) Wx[i] = cubic_gpu(i - 1 - dx);
    for (int i = 0; i < 4; i++) Wy[i] = cubic_gpu(i - 1 - dy);

    float g = grad_output[out_y * outW + out_x];
    for (int m = 0; m < 4; m++)
    for (int n = 0; n < 4; n++)
    {
        int gx = min(max(ix + n - 1, 0), inW - 1);
        int gy = min(max(iy + m - 1, 0), inH - 1);
        atomicAdd(&grad_input[gy * inW + gx], g * Wx[n] * Wy[m]);
    }
}


Mat bicubic_forward_cuda(const Mat& input, int outW, int outH)
{
    int inW = input.cols, inH = input.rows;
    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    float *d_input, *d_output;
    cudaMalloc(&d_input,  sizeof(float) * inW  * inH);
    cudaMalloc(&d_output, sizeof(float) * outW * outH);
    cudaMemcpy(d_input, input.ptr<float>(0),
               sizeof(float) * inW * inH, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float) * outW * outH);

    dim3 block(TILE, TILE);
    dim3 grid((outW + TILE - 1) / TILE, (outH + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    bicubic_forward_kernel<<<grid, block>>>(
        d_input, inW, inH, d_output, outW, outH, scaleX, scaleY);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    
    long long flops = (long long)outW * outH * (8*9 + 16*2);
    cout << "[Forward ] kernel: " << ms << " ms"
         << "  |  " << (flops/1e9)/(ms/1000.f) << " GFLOPS" << endl;

    cudaEventDestroy(t0); cudaEventDestroy(t1);

    Mat output(outH, outW, CV_32F);
    cudaMemcpy(output.ptr<float>(0), d_output,
               sizeof(float) * outW * outH, cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output);
    return output;
}

Mat bicubic_backward_cuda(const Mat& grad_out_mat, int inW, int inH)
{
    int outW = grad_out_mat.cols, outH = grad_out_mat.rows;
    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    float *d_grad_output, *d_grad_input;
    cudaMalloc(&d_grad_output, sizeof(float) * outW * outH);
    cudaMalloc(&d_grad_input,  sizeof(float) * inW  * inH);
    cudaMemcpy(d_grad_output, grad_out_mat.ptr<float>(0),
               sizeof(float) * outW * outH, cudaMemcpyHostToDevice);
    cudaMemset(d_grad_input, 0, sizeof(float) * inW * inH);

    dim3 block(TILE, TILE);
    dim3 grid((outW + TILE - 1) / TILE, (outH + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    bicubic_backward_kernel<<<grid, block>>>(
        d_grad_output, outW, outH,
        d_grad_input,  inW,  inH,
        scaleX, scaleY);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    long long flops = (long long)outW * outH * (8*9 + 16*2);
    cout << "[Backward] kernel: " << ms << " ms"
         << "  |  " << (flops/1e9)/(ms/1000.f) << " GFLOPS" << endl;

    cudaEventDestroy(t0); cudaEventDestroy(t1);

    Mat grad_input(inH, inW, CV_32F);
    cudaMemcpy(grad_input.ptr<float>(0), d_grad_input,
               sizeof(float) * inW * inH, cudaMemcpyDeviceToHost);
    cudaFree(d_grad_output); cudaFree(d_grad_input);
    return grad_input;
}


void validate(const Mat& A, const Mat& B, const string& label)
{
    double error = 0.0, max_error = 0.0;
    for (int y = 0; y < A.rows; y++)
    for (int x = 0; x < A.cols; x++)
    {
        float diff = fabs(A.at<float>(y,x) - B.at<float>(y,x));
        error    += diff;
        max_error = max(max_error, (double)diff);
    }
    cout << "\n[" << label << "]" << endl;
    cout << "  Mean  Error : " << error / (A.rows * A.cols) << endl;
    cout << "  Max   Error : " << max_error << endl;
    cout << "  Total Error : " << error << endl;
}

void save_bin(const Mat& m, const string& path)
{
    ofstream f(path, ios::binary);
    int rows = m.rows, cols = m.cols;
    f.write((char*)&rows, sizeof(int));
    f.write((char*)&cols, sizeof(int));
    f.write((char*)m.ptr<float>(0), sizeof(float) * rows * cols);
    cout << "Saved: " << path << "  [" << rows << " x " << cols << "]" << endl;
}


int main()
{
    Mat img = imread("rose.jpg");
    if (img.empty()) { cout << "Image not found!" << endl; return -1; }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int inW  = gray.cols, inH  = gray.rows;
    int outW = inW * 10,   outH = inH * 10;

    cout << "Input  : " << inH << " x " << inW << endl;
    cout << "Output : " << outH << " x " << outW << endl;

    //Mat cpu_fwd  = bicubic_forward_cpu(gray, outW, outH);
    Mat cuda_fwd = bicubic_forward_cuda(gray, outW, outH);
    //validate(cpu_fwd, cuda_fwd, "Forward: CPU vs CUDA");

    Mat grad_out = Mat::ones(outH, outW, CV_32F);
    //Mat cpu_bwd  = bicubic_backward_cpu(gray, grad_out, outW, outH);
    Mat cuda_bwd = bicubic_backward_cuda(grad_out, inW, inH);
    //validate(cpu_bwd, cuda_bwd, "Backward: CPU vs CUDA");

    //save_bin(cpu_fwd, "cpp_forward.bin");
    //save_bin(cpu_bwd, "cpp_backward.bin");

    cout << "\nDone." << endl;
    return 0;
}
