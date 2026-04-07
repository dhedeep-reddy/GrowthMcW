#include <immintrin.h>
#include <iostream>
#include <chrono>

const int N = 1024;

double A[N][N];
double B[N][N];
double C[N][N];

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

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {

            __m512d a_val = _mm512_set1_pd(A[i][k]);  // broadcast

            for (int j = 0; j < N; j += 8) {

                __m512d b_vec = _mm512_loadu_pd(&B[k][j]);
                __m512d c_vec = _mm512_loadu_pd(&C[i][j]);

                c_vec = _mm512_fmadd_pd(a_val, b_vec, c_vec);

                _mm512_storeu_pd(&C[i][j], c_vec);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();
    std::cout << "AVX-512 Time: " << time << " sec\n";
    std::cout << "C[0][0] = " << C[0][0] << "\n";

    return 0;
}