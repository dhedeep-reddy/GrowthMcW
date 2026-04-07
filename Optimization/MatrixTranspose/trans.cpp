#include <iostream>
#include <chrono>
#include <cstdlib>

constexpr size_t N = 8192;   // 8192 x 8192 matrix

double* A;
double* B;

void initialize()
{
    for (size_t i = 0; i < N * N; ++i)
        A[i] = static_cast<double>(i % 100);
}

void transpose()
{
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B[j * N + i] = A[i * N + j];
}

int main()
{
    std::cout << "Matrix Size: " << N << " x " << N << "\n";

    A = (double*) aligned_alloc(64, N * N * sizeof(double));
    B = (double*) aligned_alloc(64, N * N * sizeof(double));

    initialize();

    auto start = std::chrono::high_resolution_clock::now();

    transpose();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Time: " << duration.count() << " sec\n";

    // prevent optimization removal
    std::cout << "Check: " << B[0] << "\n";

    free(A);
    free(B);

    return 0;
}