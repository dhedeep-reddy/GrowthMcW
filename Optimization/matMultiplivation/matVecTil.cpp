#include <immintrin.h>
#include <iostream>
#include <chrono>

const int N = 1024;
const int BLOCK = 64;   // Try 32, 64, 128

alignas(64) double A[N][N];
alignas(64) double B[N][N];
alignas(64) double C[N][N];

void init() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
}

int main() {

    init();

    auto start = std::chrono::high_resolution_clock::now();

    for (int ii = 0; ii < N; ii += BLOCK)
        for (int kk = 0; kk < N; kk += BLOCK)
            for (int jj = 0; jj < N; jj += BLOCK)

                for (int i = ii; i < ii + BLOCK; i++)
                    for (int k = kk; k < kk + BLOCK; k++) {

                        __m512d a_val = _mm512_set1_pd(A[i][k]);

                        for (int j = jj; j < jj + BLOCK; j += 8) {

                            __m512d b_vec = _mm512_load_pd(&B[k][j]);
                            __m512d c_vec = _mm512_load_pd(&C[i][j]);

                            c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);

                            _mm512_store_pd(&C[i][j], c_vec);
                        }
                    }

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();
    std::cout << "AVX-512 + Tiling Time: " << time << " sec\n";
    std::cout << "C[0][0] = " << C[0][0] << "\n";

    return 0;
}