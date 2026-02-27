#include "solution.h"
#include <algorithm>
#include <array>
#include <random>
#include <cstdint>

void solution(std::vector<S>& arr) {

    // 1. Shuffle (fixed seed for benchmarking)
    static std::mt19937 g(42);
    std::shuffle(arr.begin(), arr.end(), g);

    // 2. Counting sort based on S::i
    constexpr int cntSize = maxRandom - minRandom + 1;

    std::array<int, cntSize> cnt{};
    
    // Count occurrences
    for (const auto& v : arr) {
        ++cnt[v.i];
    }

    // Prefix sum
    for (int i = 1; i < cntSize; ++i) {
        cnt[i] += cnt[i - 1];
    }

    // Build sorted array
    std::vector<S> sorted(arr.size());

    // Go backwards to maintain stability
    for (int i = arr.size() - 1; i >= 0; --i) {
        const auto& v = arr[i];
        sorted[--cnt[v.i]] = v;
    }

    // Swap instead of copy
    arr.swap(sorted);
}