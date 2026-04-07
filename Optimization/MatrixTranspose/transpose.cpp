#include <iostream>
#include <vector>

void transpose(const std::vector<std::vector<int>>& A,
               std::vector<std::vector<int>>& B,
               int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            B[j][i] = A[i][j];
}

int main() {
    int rows = 3, cols = 4;

    std::vector<std::vector<int>> A = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    std::vector<std::vector<int>> B(cols, std::vector<int>(rows));

    transpose(A, B, rows, cols);

    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j)
            std::cout << B[i][j] << " ";
        std::cout << "\n";
    }
}