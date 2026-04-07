#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void convolution2D(const Mat& input, Mat& output, const vector<vector<float>>& kernel)
{
    int K = kernel.size();
    int pad = K / 2;

    int rows = input.rows;
    int cols = input.cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float sum = 0.0;

            for (int ki = 0; ki < K; ki++)
            {
                for (int kj = 0; kj < K; kj++)
                {
                    int ni = i + ki - pad;
                    int nj = j + kj - pad;

                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols)
                    {
                        sum += input.at<uchar>(ni, nj) * kernel[ki][kj];
                    }
                }
            }

            output.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }
}

int main()
{
    // Load image
    Mat img = imread("rose.jpg");

    if (img.empty())
    {
        cout << "Error loading image\n";
        return -1;
    }

    // Convert to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Output image
    Mat output = Mat::zeros(gray.size(), gray.type());

    // Example kernel (Edge detection - Sobel-like)
    vector<vector<float>> kernel = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    convolution2D(gray, output, kernel);

    // Save result
    imwrite("output.jpg", output);

    cout << "Done! Output saved as output.jpg\n";

    return 0;
}