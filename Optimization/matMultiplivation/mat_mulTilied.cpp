#include <iostream>
#include <vector>
#include <chrono>

const int N = 1024;
const int BLOCK = 16;   // Try 16, 32, 64

std::vector<double> A(N * N);
std::vector<double> B(N * N);
std::vector<double> C(N * N);

void init() {
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }
}

int main() {
    init();

    auto start = std::chrono::high_resolution_clock::now();

    for (int ii = 0; ii < N; ii += BLOCK)
        for (int jj = 0; jj < N; jj += BLOCK)
            for (int kk = 0; kk < N; kk += BLOCK)

                for (int i = ii; i < ii + BLOCK; i++)
                    for (int j = jj; j < jj + BLOCK; j++)
                        for (int k = kk; k < kk + BLOCK; k++)
                            C[i*N + j] += A[i*N + k] * B[k*N + j];

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();
    std::cout << "Tiled Time: " << time << " seconds\n";
    std::cout << "C[0] = " << C[0] << "\n";

    return 0;
}