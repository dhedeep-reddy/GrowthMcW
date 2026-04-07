#include <iostream>
#include <vector>
using namespace std;

// Function to perform convolution
void convolution2D(const vector<vector<float>>& input,
                   vector<vector<float>>& output,
                   const vector<vector<float>>& kernel,
                   int N, int M, int K)
{
    int pad = K / 2;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float sum = 0.0;

            for (int ki = 0; ki < K; ki++)
            {
                for (int kj = 0; kj < K; kj++)
                {
                    int ni = i + ki - pad;
                    int nj = j + kj - pad;

                    // Boundary check (zero padding)
                    if (ni >= 0 && ni < N && nj >= 0 && nj < M)
                    {
                        sum += input[ni][nj] * kernel[ki][kj];
                    }
                }
            }

            output[i][j] = sum;
        }
    }
}

int main()
{
    int N = 5, M = 5;   // Input size
    int K = 3;          // Kernel size

    // Input matrix
    vector<vector<float>> input = {
        {1, 2, 3, 4, 5},
        {5, 6, 7, 8, 9},
        {9, 1, 2, 3, 4},
        {4, 5, 6, 7, 8},
        {8, 9, 1, 2, 3}
    };

    // Example kernel (Sharpen)
    vector<vector<float>> kernel = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    vector<vector<float>> output(N, vector<float>(M, 0));

    convolution2D(input, output, kernel, N, M, K);

    // Print output
    cout << "Output:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            cout << output[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}