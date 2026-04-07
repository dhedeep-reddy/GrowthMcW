#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

long long sum_parallel(vector<int>& arr) {
    long long s = 0;

    #pragma omp parallel for reduction(+:s) schedule(static)
    for (size_t i = 0; i < arr.size(); i++) {
        int x = arr[i];

        x = x * 13;
        x ^= 0x5a5a5a5a;
        x = x * 7;
        x += 12345;
        x = x * 3;
        x ^= 0xdeadbeef;

        s += x;
    }

    return s;
}

int main() {
    const size_t N = 100000000;

    vector<int> arr(N, 1);

    double start = omp_get_wtime();
    long long result = sum_parallel(arr);
    double end = omp_get_wtime();

    cout << "Result: " << result << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    return 0;
}