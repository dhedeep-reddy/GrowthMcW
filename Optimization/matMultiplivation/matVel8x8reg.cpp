#include <immintrin.h>
#include <iostream>
#include <chrono>

const int N = 1024;
const int BLOCK = 32;   // L1-friendly (important)

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
        for (int jj = 0; jj < N; jj += BLOCK)
            for (int kk = 0; kk < N; kk += BLOCK)

                for (int i = ii; i < ii + BLOCK; i += 8)
                    for (int j = jj; j < jj + BLOCK; j += 8) {

                        __m512d c0 = _mm512_load_pd(&C[i+0][j]);
                        __m512d c1 = _mm512_load_pd(&C[i+1][j]);
                        __m512d c2 = _mm512_load_pd(&C[i+2][j]);
                        __m512d c3 = _mm512_load_pd(&C[i+3][j]);
                        __m512d c4 = _mm512_load_pd(&C[i+4][j]);
                        __m512d c5 = _mm512_load_pd(&C[i+5][j]);
                        __m512d c6 = _mm512_load_pd(&C[i+6][j]);
                        __m512d c7 = _mm512_load_pd(&C[i+7][j]);

                        for (int k = kk; k < kk + BLOCK; k++) {

                            __m512d b_vec = _mm512_load_pd(&B[k][j]);

                            __m512d a0 = _mm512_set1_pd(A[i+0][k]);
                            __m512d a1 = _mm512_set1_pd(A[i+1][k]);
                            __m512d a2 = _mm512_set1_pd(A[i+2][k]);
                            __m512d a3 = _mm512_set1_pd(A[i+3][k]);
                            __m512d a4 = _mm512_set1_pd(A[i+4][k]);
                            __m512d a5 = _mm512_set1_pd(A[i+5][k]);
                            __m512d a6 = _mm512_set1_pd(A[i+6][k]);
                            __m512d a7 = _mm512_set1_pd(A[i+7][k]);

                            c0 = _mm512_fmadd_pd(a0, b_vec, c0);
                            c1 = _mm512_fmadd_pd(a1, b_vec, c1);
                            c2 = _mm512_fmadd_pd(a2, b_vec, c2);
                            c3 = _mm512_fmadd_pd(a3, b_vec, c3);
                            c4 = _mm512_fmadd_pd(a4, b_vec, c4);
                            c5 = _mm512_fmadd_pd(a5, b_vec, c5);
                            c6 = _mm512_fmadd_pd(a6, b_vec, c6);
                            c7 = _mm512_fmadd_pd(a7, b_vec, c7);
                        }

                        _mm512_store_pd(&C[i+0][j], c0);
                        _mm512_store_pd(&C[i+1][j], c1);
                        _mm512_store_pd(&C[i+2][j], c2);
                        _mm512_store_pd(&C[i+3][j], c3);
                        _mm512_store_pd(&C[i+4][j], c4);
                        _mm512_store_pd(&C[i+5][j], c5);
                        _mm512_store_pd(&C[i+6][j], c6);
                        _mm512_store_pd(&C[i+7][j], c7);
                    }

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();

    double gflops = (2.0 * N * N * N) / time / 1e9;

    std::cout << "Optimized Time: " << time << " sec\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "C[0][0] = " << C[0][0] << "\n";

    return 0;
}