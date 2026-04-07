#include <iostream>
#include <chrono>
#include <cstdlib>

constexpr size_t N = 8192;
constexpr size_t BLOCK = 64;   

double* A;
double* B;

void initialize()
{
    for (size_t i = 0; i < N * N; ++i)
        A[i] = static_cast<double>(i % 100);
}

void transpose_blocked()
{
    for (size_t ii = 0; ii < N; ii += BLOCK)
    {
        for (size_t jj = 0; jj < N; jj += BLOCK)
        {
            for (size_t i = ii; i < ii + BLOCK; ++i)
            {
                for (size_t j = jj; j < jj + BLOCK; ++j)
                {
                    B[j * N + i] = A[i * N + j];
                }
            }
        }
    }
}

int main()
{
    std::cout << "Matrix Size: " << N << " x " << N << "\n";

    A = (double*) aligned_alloc(64, N * N * sizeof(double));
    B = (double*) aligned_alloc(64, N * N * sizeof(double));

    initialize();

    auto start = std::chrono::high_resolution_clock::now();

    transpose_blocked();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Time: " << duration.count() << " sec\n";
    std::cout << "Check: " << B[0] << "\n";

    free(A);
    free(B);

    return 0;
}