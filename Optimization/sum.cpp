#include <iostream>
using namespace std;

// Function to calculate sum of array elements
int sum(int* arr, int n) {
    int s = 0;
    for(int i = 0; i < n; i++)
        s += arr[i];
    return s;
}

int main() {
    int n=5;

    int arr[n]={1,2,3,4,5}; 


    int result = sum(arr, n);

    cout << "Sum of elements = " << result << endl;

    return 0;
}