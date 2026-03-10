# Parallel-programming-model

## Task 1 — Sinus

An array of 10⁷ elements is filled with sine values (one period over the entire length). The sum of elements is calculated and output to the terminal.

### Choosing array type at build time

The array element type is set by the CMake option `USE_DOUBLE`:

| Array type | Configuration command |
|------------|----------------------|
| **float** (default) | `cmake -B build` or `cmake -B build -DUSE_DOUBLE=OFF` |
| **double** | `cmake -B build -DUSE_DOUBLE=ON` |

### Build and run

From the repository root:

```bash
# float (default)
cmake -B build
cmake --build build
./build/task1/sinus

# double
cmake -B build -DUSE_DOUBLE=ON
cmake --build build
./build/task1/sinus
```

From the `task1` folder:

```bash
cd task1
cmake -B build -DUSE_DOUBLE=OFF  # float
cmake -B build -DUSE_DOUBLE=ON   # double
cmake --build build
./build/sinus
```

### Results (terminal output)

After building and running, the program outputs the array type, number of elements, and sum. Example:

**float** (expected: sum close to 0, larger error than double):
```
Sum: -0.0956784
```

**double** (expected: sum close to 0, smaller error than float):
```
Sum: 6.82116e-07
```
