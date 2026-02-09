#include <iostream>
#include <vector>
#include <cmath>

// change vertor type
#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

int main() {
    int n = 10000000;
    std::vector<real> arr(n);
    real sum = 0.0;
    double pi = 3.14159265;

    for (int i = 0; i < n; i++) {
        arr[i] = (real)sin(2 * pi * i / n);
        sum += arr[i];
    }

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}
