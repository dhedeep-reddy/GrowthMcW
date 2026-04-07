#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
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

Mat bicubic_manual(const Mat& input, int outW, int outH)
{
    int inW = input.cols;
    int inH = input.rows;

    Mat output(outH, outW, CV_32F);

    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;

    for(int y = 0; y < outH; y++)
    {
        for(int x = 0; x < outW; x++)
        {
            float in_x = (x + 0.5f) * scaleX - 0.5f;
            float in_y = (y + 0.5f) * scaleY - 0.5f;

            in_x = max(0.0f, min(in_x, (float)(inW - 1)));
            in_y = max(0.0f, min(in_y, (float)(inH - 1)));

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

                    float pixel = input.at<float>(py, px);
                    sum += pixel * Wx[n+1] * Wy[m+1];
                }
            }

            output.at<float>(y, x) = sum;
        }
    }

    return output;
}


Mat bicubic_opencv(const Mat& input, int outW, int outH)
{
    Mat output;
    resize(input, output, Size(outW, outH), 0, 0, INTER_CUBIC);
    return output;
}


void validate(const Mat& A, const Mat& B)
{
    double error = 0.0;
    double max_error = 0.0;

    for(int y = 0; y < A.rows; y++)
    {
        for(int x = 0; x < A.cols; x++)
        {
            float diff = abs(A.at<float>(y,x) - B.at<float>(y,x));
            error += diff;
            max_error = max(max_error, (double)diff);
        }
    }

    cout << "\nTotal Error: " << error << endl;
    cout << "Max Pixel Error: " << max_error << endl;
    cout << "Mean Error: " << error / (A.rows * A.cols) << endl;
}

int main()
{
    Mat img = imread("rose.jpg");

    if(img.empty())
    {
        cout << "Image not found!" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    gray.convertTo(gray, CV_32F);

    int outW = gray.cols * 2;
    int outH = gray.rows * 2;

    Mat opencv_resized = bicubic_opencv(gray, outW, outH);

    Mat manual_resized = bicubic_manual(gray, outW, outH);

    validate(opencv_resized, manual_resized);

    opencv_resized.convertTo(opencv_resized, CV_8U);
    manual_resized.convertTo(manual_resized, CV_8U);

    imwrite("opencv_bicubic.jpg", opencv_resized);
    imwrite("manual_bicubic.jpg", manual_resized);

    cout << "\nImages saved:\n";
    cout << "opencv_bicubic.jpg\n";
    cout << "manual_bicubic.jpg\n";

    return 0;
}