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


__device__ __forceinline__ float cubic_gpu(float x)
{
    const float a = -0.75f;
    x = fabsf(x);
    float x2 = x * x;
    float x3 = x2 * x;
    float t1 = __fmaf_rn(a + 2.f, x3, __fmaf_rn(-(a + 3.f), x2, 1.f));
    float t2 = __fmaf_rn(a, x3, __fmaf_rn(-5.f*a, x2, __fmaf_rn(8.f*a, x, -4.f*a)));
    float r  = (x <= 1.f) ? t1 : t2;
    return     (x <  2.f) ? r  : 0.f;
}


__device__ __forceinline__ float4 cubic4_gpu(float d)
{
    return make_float4(
        cubic_gpu(-1.f - d),
        cubic_gpu( 0.f - d),
        cubic_gpu( 1.f - d),
        cubic_gpu( 2.f - d)
    );
}


// CPU Forward 
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


// CPU Backward 
Mat bicubic_backward_cpu(const Mat& input, const Mat& grad_output,
                            int outW, int outH){
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


// Defines
#define TILE        16
#define MAX_SCALE   2

#define SMEM_W      (TILE * 2 * MAX_SCALE + 4)
#define SMEM_H      (TILE * 2 * MAX_SCALE + 4)
#define SMEM_WP     (SMEM_W + 1)               

// Backward: block covers TILE outputs (1x1 per thread)
#define SMEM_W_BWD  (TILE * MAX_SCALE + 4)
#define SMEM_H_BWD  (TILE * MAX_SCALE + 4)
#define SMEM_WP_BWD (SMEM_W_BWD + 1)           

#define WARPS_PER_BLOCK  ((TILE * TILE) / 32)


// CUDA Forward Kernel
__global__ void bicubic_forward_kernel(
    const float* __restrict__ input,  int inW,  int inH,
          float* __restrict__ output, int outW, int outH,
    float scaleX, float scaleY)
{
    __shared__ float smemA[SMEM_H][SMEM_WP];   // double buffer
    __shared__ float smemB[SMEM_H][SMEM_WP];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * TILE + tx;

    
    int base_ox = blockIdx.x * TILE * 2 + tx * 2;
    int base_oy = blockIdx.y * TILE * 2 + ty * 2;


    float in_x0f = (blockIdx.x * TILE * 2 + 0.5f) * scaleX - 0.5f;
    float in_y0f = (blockIdx.y * TILE * 2 + 0.5f) * scaleY - 0.5f;
    int ix0 = (int)floorf(in_x0f) - 1;
    int iy0 = (int)floorf(in_y0f) - 1;


    int smem_w = min((int)ceilf(TILE * 2 * scaleX) + 4, SMEM_W);
    int smem_h = min((int)ceilf(TILE * 2 * scaleY) + 4, SMEM_H);
    int total  = smem_w * smem_h;


    for (int i = tid; i < total; i += TILE * TILE)
    {
        int sy = i / smem_w, sx = i % smem_w;
        int gx = min(max(ix0 + sx, 0), inW - 1);
        int gy = min(max(iy0 + sy, 0), inH - 1);
        smemA[sy][sx] = __ldg(&input[gy * inW + gx]);  // OPT 3
    }
    __syncthreads();

    float (*curBuf)[SMEM_WP] = smemA;
    float (*nxtBuf)[SMEM_WP] = smemB;

    float results[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

    for (int rr = 0; rr < 2; rr++)
    for (int rc = 0; rc < 2; rc++)
    {
        int out_x = base_ox + rc;
        int out_y = base_oy + rr;
        if (out_x >= outW || out_y >= outH) continue;

        float in_x = (out_x + 0.5f) * scaleX - 0.5f;
        float in_y = (out_y + 0.5f) * scaleY - 0.5f;
        in_x = fmaxf(0.f, fminf(in_x, (float)(inW - 1)));
        in_y = fmaxf(0.f, fminf(in_y, (float)(inH - 1)));

        int ix = (int)floorf(in_x);
        int iy = (int)floorf(in_y);
        float dx = in_x - ix;
        float dy = in_y - iy;

        int lx = ix - ix0;
        int ly = iy - iy0;


        float4 Wx4 = cubic4_gpu(dx);
        float4 Wy4 = cubic4_gpu(dy);
        float  Wxv[4] = {Wx4.x, Wx4.y, Wx4.z, Wx4.w};
        float  Wyv[4] = {Wy4.x, Wy4.y, Wy4.z, Wy4.w};



        float sum = 0.f;
        #pragma unroll 
        for (int m = 0; m < 4; m++)
        {
            #pragma unroll 4
            for (int n = 0; n < 4; n++)
            {
                int sy = ly + m - 1;
                int sx = lx + n - 1;
                float val;
                if (sy >= 0 && sy < smem_h && sx >= 0 && sx < smem_w)
                    val = curBuf[sy][sx];
                else {
                    int gx = min(max(ix + n - 1, 0), inW - 1);
                    int gy = min(max(iy + m - 1, 0), inH - 1);
                    val = __ldg(&input[gy * inW + gx]);
                }
                sum = __fmaf_rn(val, Wxv[n] * Wyv[m], sum);
            }
        }

        results[rr][rc] = sum;
    }

    // load next
    int next_ix0 = ix0 + (int)ceilf(TILE * 2 * scaleX);
    for (int i = tid; i < total; i += TILE * TILE)
    {
        int sy = i / smem_w, sx = i % smem_w;
        int gx = min(max(next_ix0 + sx, 0), inW - 1);
        int gy = min(max(iy0     + sy, 0), inH - 1);
        nxtBuf[sy][sx] = __ldg(&input[gy * inW + gx]);  
    }

    // write 2x2 results
    #pragma unroll 2
    for (int rr = 0; rr < 2; rr++)
    #pragma unroll 2
    for (int rc = 0; rc < 2; rc++)
    {
        int out_x = base_ox + rc;
        int out_y = base_oy + rr;
        if (out_x < outW && out_y < outH)
            output[out_y * outW + out_x] = results[rr][rc];
    }
}


// CUDA Backward Kernel
__global__ void bicubic_backward_kernel(
    const float* __restrict__ grad_output, int outW, int outH,
          float* __restrict__ grad_input,  int inW,  int inH,
    float scaleX, float scaleY)
{
    __shared__ float smem[SMEM_H_BWD][SMEM_WP_BWD];  

    int tx   = threadIdx.x, ty = threadIdx.y;
    int tid  = ty * TILE + tx;

    int out_x = blockIdx.x * TILE + tx;
    int out_y = blockIdx.y * TILE + ty;

    int smem_w = min((int)ceilf(TILE * scaleX) + 4, SMEM_W_BWD);
    int smem_h = min((int)ceilf(TILE * scaleY) + 4, SMEM_H_BWD);
    int total  = smem_w * smem_h;

    // zero smem
    for (int i = tid; i < total; i += TILE * TILE)
        smem[i / smem_w][i % smem_w] = 0.f;
    __syncthreads();

    float in_x0f = (blockIdx.x * TILE + 0.5f) * scaleX - 0.5f;
    float in_y0f = (blockIdx.y * TILE + 0.5f) * scaleY - 0.5f;
    int ix0 = (int)floorf(in_x0f) - 1;
    int iy0 = (int)floorf(in_y0f) - 1;

    if (out_x < outW && out_y < outH)
    {

        float in_x = (out_x + 0.5f) * scaleX - 0.5f;
        float in_y = (out_y + 0.5f) * scaleY - 0.5f;
        in_x = fmaxf(0.f, fminf(in_x, (float)(inW - 1)));
        in_y = fmaxf(0.f, fminf(in_y, (float)(inH - 1)));

        int ix = (int)floorf(in_x);
        int iy = (int)floorf(in_y);
        float dx = in_x - ix;
        float dy = in_y - iy;

        int lx = ix - ix0;
        int ly = iy - iy0;

        float4 Wx4 = cubic4_gpu(dx);
        float4 Wy4 = cubic4_gpu(dy);
        float  Wxv[4] = {Wx4.x, Wx4.y, Wx4.z, Wx4.w};
        float  Wyv[4] = {Wy4.x, Wy4.y, Wy4.z, Wy4.w};

        float g = __ldg(&grad_output[out_y * outW + out_x]);

        int lane = tid & 31;          
        // int warp = tid >> 5;          

        #pragma unroll 4
        for (int m = 0; m < 4; m++) {
            
            for (int n = 0; n < 4; n++) {
                int sy = ly + m - 1;
                int sx = lx + n - 1;
                float contrib = __fmaf_rn(g, Wxv[n] * Wyv[m], 0.f);

                if (sy >= 0 && sy < smem_h && sx >= 0 && sx < smem_w)
                {
                   
                    float val = contrib;
                    val += __shfl_xor_sync(0xffffffff, val, 16);
                    val += __shfl_xor_sync(0xffffffff, val,  8);
                    val += __shfl_xor_sync(0xffffffff, val,  4);
                    val += __shfl_xor_sync(0xffffffff, val,  2);
                    val += __shfl_xor_sync(0xffffffff, val,  1);

                    
                    if (lane == 0)
                        atomicAdd(&smem[sy][sx], val);
                }
                else {
                   
                    int gx = min(max(ix + n - 1, 0), inW - 1);
                    int gy = min(max(iy + m - 1, 0), inH - 1);
                    atomicAdd(&grad_input[gy * inW + gx], contrib);
                }
            }
        }
    }
    __syncthreads();

        // smem to global
    for (int i = tid; i < total; i += TILE * TILE)
    {
        int sy = i / smem_w;
        int sx = i % smem_w;
        float v = smem[sy][sx];
        if (v != 0.f)
        {
            int gx = min(max(ix0 + sx, 0), inW - 1);
            int gy = min(max(iy0 + sy, 0), inH - 1);
            atomicAdd(&grad_input[gy * inW + gx], v);
        }
    }
    
}


// Host wrappers
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
    dim3 grid((outW + TILE*2 - 1) / (TILE*2),
              (outH + TILE*2 - 1) / (TILE*2));

    bicubic_forward_kernel<<<grid, block>>>(
        d_input, inW, inH, d_output, outW, outH, scaleX, scaleY);
    cudaDeviceSynchronize();

    Mat output(outH, outW, CV_32F);
    cudaMemcpy(output.ptr<float>(0), d_output,
               sizeof(float) * outW * outH, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
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

    bicubic_backward_kernel<<<grid, block>>>(
        d_grad_output, outW, outH,
        d_grad_input,  inW,  inH,
        scaleX, scaleY);
    cudaDeviceSynchronize();

    Mat grad_input(inH, inW, CV_32F);
    cudaMemcpy(grad_input.ptr<float>(0), d_grad_input,
               sizeof(float) * inW * inH, cudaMemcpyDeviceToHost);

    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    return grad_input;
}


// Utilities 
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


// Main
int main()
{
    Mat img = imread("rose.jpg");
    if (img.empty()) { cout << "Image not found!" << endl; return -1; }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int inW  = gray.cols, inH  = gray.rows;
    int outW = inW * 2,   outH = inH * 2;

    cout << "Input  : " << inH << " x " << inW << endl;
    cout << "Output : " << outH << " x " << outW << endl;

    Mat cpu_fwd  = bicubic_forward_cpu(gray, outW, outH);
    Mat cuda_fwd = bicubic_forward_cuda(gray, outW, outH);
    validate(cpu_fwd, cuda_fwd, "Forward: CPU vs CUDA");

    Mat grad_out = Mat::ones(outH, outW, CV_32F);
    Mat cpu_bwd  = bicubic_backward_cpu(gray, grad_out, outW, outH);
    Mat cuda_bwd = bicubic_backward_cuda(grad_out, inW, inH);
    validate(cpu_bwd, cuda_bwd, "Backward: CPU vs CUDA");

    save_bin(cpu_fwd, "cpp_forward.bin");
    save_bin(cpu_bwd, "cpp_backward.bin");

    cout << "\nDone." << endl;
    return 0;
}