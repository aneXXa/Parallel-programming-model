/*
 * sinus.cpp
 * Fill array with one period of sine, compute sum (float or double at build time).
 */

#include <iostream>
#include <vector>
#include <cmath>

#ifdef USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

constexpr size_t N = 10'000'000;  // 10^7

int main() {
    const double pi = 3.14159265358979323846;
    std::vector<real_t> arr(N);

    for (size_t i = 0; i < N; ++i) {
        arr[i] = static_cast<real_t>(std::sin(2.0 * pi * static_cast<double>(i) / static_cast<double>(N)));
    }

    real_t sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += arr[i];
    }

    std::cout << "Array type: " << (sizeof(real_t) == 8 ? "double" : "float") << std::endl;
    std::cout << "Elements:   " << N << std::endl;
    std::cout << "Sum:        " << sum << std::endl;

    return 0;
}
