#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <cstdlib>
using namespace std;
using namespace cv;

float cubic(float x)
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

void compute_weights(float dx, float dy, float Wx[4], float Wy[4])
{
    for (int i = -1; i <= 2; i++) Wx[i+1] = cubic(i - dx);
    for (int i = -1; i <= 2; i++) Wy[i+1] = cubic(dy - i);
}

Mat bicubic_forward(const Mat& input, int outW, int outH)
{
    int inW = input.cols;
    int inH = input.rows;

    Mat output(outH, outW, CV_32F);

    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    for (int y = 0; y < outH; y++)
    {
        for (int x = 0; x < outW; x++)
        {
            float in_x = (x + 0.5f) * scaleX - 0.5f;
            float in_y = (y + 0.5f) * scaleY - 0.5f;

            in_x = max(0.0f, min(in_x, (float)(inW - 1)));
            in_y = max(0.0f, min(in_y, (float)(inH - 1)));

            int ix = (int)floor(in_x);
            int iy = (int)floor(in_y);

            float dx = in_x - ix;
            float dy = in_y - iy;

            float Wx[4], Wy[4];
            compute_weights(dx, dy, Wx, Wy);

            float sum = 0.0f;
            for (int m = -1; m <= 2; m++)
                for (int n = -1; n <= 2; n++)
                {
                    int px = min(max(ix + n, 0), inW - 1);
                    int py = min(max(iy + m, 0), inH - 1);
                    sum += input.at<float>(py, px) * Wx[n+1] * Wy[m+1];
                }

            output.at<float>(y, x) = sum;
        }
    }

    return output;
}

Mat bicubic_backward(const Mat& input, const Mat& grad_output, int outW, int outH)
{
    int inW = input.cols;
    int inH = input.rows;

    Mat grad_input = Mat::zeros(inH, inW, CV_32F);

    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    for (int y = 0; y < outH; y++)
    {
        for (int x = 0; x < outW; x++)
        {
            float in_x = (x + 0.5f) * scaleX - 0.5f;
            float in_y = (y + 0.5f) * scaleY - 0.5f;

            in_x = max(0.0f, min(in_x, (float)(inW - 1)));
            in_y = max(0.0f, min(in_y, (float)(inH - 1)));

            int ix = (int)floor(in_x);
            int iy = (int)floor(in_y);

            float dx = in_x - ix;
            float dy = in_y - iy;

            float Wx[4], Wy[4];
            compute_weights(dx, dy, Wx, Wy);

            float g = grad_output.at<float>(y, x);

            for (int m = -1; m <= 2; m++)
                for (int n = -1; n <= 2; n++)
                {
                    int px = min(max(ix + n, 0), inW - 1);
                    int py = min(max(iy + m, 0), inH - 1);
                    grad_input.at<float>(py, px) += g * Wx[n+1] * Wy[m+1];
                }
        }
    }

    return grad_input;
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

void validate(const Mat& A, const Mat& B, const string& label)
{
    double error = 0.0;
    double max_error = 0.0;

    for (int y = 0; y < A.rows; y++)
        for (int x = 0; x < A.cols; x++)
        {
            float diff = fabs(A.at<float>(y, x) - B.at<float>(y, x));
            error += diff;
            max_error = max(max_error, (double)diff);
        }

    cout << "\n[" << label << "]" << endl;
    cout << "  Mean Error : " << error / (A.rows * A.cols) << endl;
    cout << "  Max  Error : " << max_error << endl;
    cout << "  Total Error: " << error << endl;
}

int main()
{
    Mat img = imread("rose.jpg");
    if (img.empty()) { cout << "Image not found!" << endl; return -1; }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int inW  = gray.cols;
    int inH  = gray.rows;
    int outW = inW * 2;
    int outH = inH * 2;

    cout << "Input  : " << inH << " x " << inW << endl;
    cout << "Output : " << outH << " x " << outW << endl;

    // Forward
    Mat manual_out = bicubic_forward(gray, outW, outH);

    // Generate random grad_output in range -1 to 1 
    srand(42);
    Mat grad_out(outH, outW, CV_32F);
    for (int y = 0; y < outH; y++)
        for (int x = 0; x < outW; x++)
            grad_out.at<float>(y, x) = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    // Backward 
    Mat grad_in = bicubic_backward(gray, grad_out, outW, outH);

    cout << "\n[Backward]" << endl;
    cout << "  grad_input shape : " << grad_in.rows << " x " << grad_in.cols << endl;

    double mn, mx;
    minMaxLoc(grad_in, &mn, &mx);
    cout << "  grad_input range : [" << mn << ", " << mx << "]" << endl;

    Scalar mean_grad = mean(grad_in);
    cout << "  grad_input mean  : " << mean_grad[0] << endl;

    
    save_bin(manual_out, "cpp_forward.bin");
    save_bin(grad_out,   "grad_output.bin");  
    save_bin(grad_in,    "cpp_backward.bin");

    cout << "\nDone. Run back.py to compare with PyTorch." << endl;

    return 0;
}