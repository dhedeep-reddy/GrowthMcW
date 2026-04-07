#include <iostream>
#include <vector>
#include <chrono>

const int N = 1024;

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

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();
    std::cout << "Naive Time: " << time << " seconds\n";
    std::cout << "C[0] = " << C[0] << "\n";

    return 0;
}