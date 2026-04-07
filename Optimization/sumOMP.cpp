#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

long long sum_parallel(vector<int>& arr) {
    long long s = 0;

    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < arr.size(); i++) {
        s += arr[i];
    }

    return s;
}

int main() {
    const size_t N = 500000000;  // 100 million (~381 MiB)

    vector<int> arr(N, 1);

    double start = omp_get_wtime();

    long long result = sum_parallel(arr);

    double end = omp_get_wtime();

    cout << "Result: " << result << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    return 0;
}