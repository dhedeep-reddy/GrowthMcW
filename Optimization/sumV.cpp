#include <iostream>
#include <vector>

using namespace std;

long long sum(vector<int>& arr) {
    long long s = 0;
    for (size_t i = 0; i < arr.size(); i++)
        s += arr[i];
    return s;
}

int main() {
    const size_t N = 100000000;  // 100 million
    vector<int> arr(N, 1);

    long long result = sum(arr);

    cout << result << endl;
    return 0;
}