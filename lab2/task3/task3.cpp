#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>

#include <omp.h>
#include <stdio.h>
#include <time.h>

static const int MAX_ITER = 1000;
static const double TOL = 1e-10;

static const std::size_t N_EXPERIMENT = 15000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// ---------- matrix generation ----------
// A[i][i] = 2.0, A[i][j] = 1.0 for i != j; b[i] = N + 1 (exact solution x = 1)
void generate_system(std::vector<double> &a, std::vector<double> &b, std::size_t n)
{
    a.resize(n * n);
    b.resize(n);

    for (std::size_t i = 0; i < n; i++)
    {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; j++)
        {
            if (i == j)
                a[i * n + j] = 2.0;
            else
                a[i * n + j] = 1.0;
            row_sum += a[i * n + j];
        }
        // For x = 1  b[i] = sum_j A[i][j] * 1 = N + 1
        b[i] = row_sum;
    }
}

// ---------- helper: high‑resolution timer wrapper for solvers ----------
double run_solver(int (*solver)(const double *, const double *, double *, std::size_t, int, double),
                  const double *a, const double *b, double *x, std::size_t n,
                  int max_iter, double tol, int nthreads)
{
    omp_set_num_threads(nthreads);
    double t = cpuSecond();
    int iters = solver(a, b, x, n, max_iter, tol);
    t = cpuSecond() - t;
    if (iters < 0)
        std::cerr << "Warning: solver did not converge (threads=" << nthreads << ")\n";
    return t;
}

// ---------- serial Jacobi (baseline) ----------
int solve_jacobi_serial(const double *a, const double *b, double *x, std::size_t n,
                        int max_iter, double tol)
{
    if (n == 0 || a == nullptr || b == nullptr || x == nullptr)
        return -1;
    for (std::size_t i = 0; i < n; i++)
        if (std::abs(a[i * n + i]) < 1e-15)
            return -1;

    // Richardson iteration parameter; chosen conservatively based on n
    double tau = 1.0 / (2.0 * static_cast<double>(n));

    std::vector<double> x_old(n, 0.0);
    int iter;

    for (iter = 0; iter < max_iter; iter++)
    {
        double res_norm = 0.0;
        for (std::size_t i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; j++)
                sum += a[i * n + j] * x_old[j];

            double ri = b[i] - sum;
            x[i] = x_old[i] + tau * ri;
            res_norm += ri * ri;
        }

        res_norm = std::sqrt(res_norm);
        if (res_norm <= tol)
            return iter + 1;

        std::memcpy(x_old.data(), x, n * sizeof(double));
    }
    return -1; // not converged
}

// ---------- parallel Jacobi, Variant 1: separate parallel for for each loop ----------
int solve_jacobi_parallel_blocks(const double *a, const double *b, double *x, std::size_t n,
                                 int max_iter, double tol)
{
    if (n == 0 || a == nullptr || b == nullptr || x == nullptr)
        return -1;
    for (std::size_t i = 0; i < n; i++)
        if (std::abs(a[i * n + i]) < 1e-15)
            return -1;

    double tau = 1.0 / (2.0 * static_cast<double>(n));

    std::vector<double> x_old(n, 0.0);
    int iter;

    for (iter = 0; iter < max_iter; iter++)
    {
        double res_norm = 0.0;

#pragma omp parallel for
        for (std::size_t i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; j++)
                sum += a[i * n + j] * x_old[j];

            double ri = b[i] - sum;
            x[i] = x_old[i] + tau * ri;
        }

#pragma omp parallel for reduction(+ : res_norm)
        for (std::size_t i = 0; i < n; i++)
        {
            double ri = b[i];
            for (std::size_t j = 0; j < n; j++)
                ri -= a[i * n + j] * x_old[j];
            res_norm += ri * ri;
        }

        res_norm = std::sqrt(res_norm);
        if (res_norm <= tol)
            return iter + 1;

        std::memcpy(x_old.data(), x, n * sizeof(double));
    }
    return -1;
}

// ---------- parallel Jacobi, Variant 2: one #pragma omp parallel for whole algorithm ----------
int solve_jacobi_parallel_whole(const double *a, const double *b, double *x, std::size_t n,
                                int max_iter, double tol)
{
    if (n == 0 || a == nullptr || b == nullptr || x == nullptr)
        return -1;
    for (std::size_t i = 0; i < n; i++)
        if (std::abs(a[i * n + i]) < 1e-15)
            return -1;

    double tau = 1.0 / (2.0 * static_cast<double>(n));

    std::vector<double> x_old(n, 0.0);
    int converged = 0;
    int last_iter = -1;
    double res_norm = 0.0;
    double local_res = 0.0;

#pragma omp parallel shared(x_old, x, converged, last_iter, res_norm, local_res)
    {
        for (int iter = 0; iter < max_iter && !converged; iter++)
        {
#pragma omp for
            for (std::size_t i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (std::size_t j = 0; j < n; j++)
                    sum += a[i * n + j] * x_old[j];

                double ri = b[i] - sum;
                x[i] = x_old[i] + tau * ri;
            }

#pragma omp single
            local_res = 0.0;

#pragma omp for reduction(+ : local_res)
            for (std::size_t i = 0; i < n; i++)
            {
                double ri = b[i];
                for (std::size_t j = 0; j < n; j++)
                    ri -= a[i * n + j] * x_old[j];
                local_res += ri * ri;
            }

#pragma omp single
            {
                res_norm = std::sqrt(local_res);
                if (res_norm <= tol)
                {
                    converged = 1;
                    last_iter = iter + 1;
                }
                if (!converged)
                    std::memcpy(x_old.data(), x, n * sizeof(double));
            }
        }
    }

    return converged ? last_iter : -1;
}

// ---------- CSV output (instead of gnuplot scripts) ----------
void write_speedup_csv_jacobi(std::size_t n,
                              const std::vector<int> &threads,
                              double T_serial,
                              const std::vector<double> &T_blocks,
                              const std::vector<double> &T_whole)
{
    if (threads.empty() ||
        T_blocks.size() != threads.size() ||
        T_whole.size() != threads.size())
        return;

    std::ostringstream name_suffix;
    name_suffix << n;
    std::string csv_filename = "speedup_jacobi_" + name_suffix.str() + ".csv";

    std::ofstream csv(csv_filename.c_str());
    if (!csv)
    {
        std::cerr << "Cannot open " << csv_filename << " for writing\n";
        return;
    }

    csv << "threads,T_serial,T_blocks,S_blocks,T_whole,S_whole\n";
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        double S_blocks = (T_blocks[i] > 0.0) ? (T_serial / T_blocks[i]) : 0.0;
        double S_whole = (T_whole[i] > 0.0) ? (T_serial / T_whole[i]) : 0.0;
        csv << threads[i] << ","
            << T_serial << ","
            << T_blocks[i] << "," << S_blocks << ","
            << T_whole[i] << "," << S_whole << "\n";
    }
    csv.close();
}

// ---------- experiment driver ----------
void run_experiments()
{
    const std::size_t sizes[] = {N_EXPERIMENT};
    const int threads_list[] = {1, 2, 4, 8, 16, 20, 40};
    const std::size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const std::size_t n_threads = sizeof(threads_list) / sizeof(threads_list[0]);

    std::cout << std::fixed << std::setprecision(6);

    for (std::size_t s = 0; s < n_sizes; ++s)
    {
        std::size_t n = sizes[s];
        std::cout << "\n========== Jacobi: matrix size n = " << n << " ==========\n";

        std::vector<double> a, b, x_serial(n, 0.0);
        generate_system(a, b, n);

        std::fill(x_serial.begin(), x_serial.end(), 0.0);
        double T_serial = run_solver(solve_jacobi_serial,
                                     a.data(), b.data(), x_serial.data(),
                                     n, MAX_ITER, TOL, 1);
        std::cout << "T_serial = " << T_serial << " sec\n\n";

        std::cout << std::setw(8) << "threads"
                  << std::setw(15) << "T_blocks"
                  << std::setw(15) << "S_blocks"
                  << std::setw(15) << "T_whole"
                  << std::setw(15) << "S_whole" << "\n";

        std::vector<int> used_threads;
        std::vector<double> T_blocks_vec;
        std::vector<double> T_whole_vec;

        for (std::size_t t = 0; t < n_threads; ++t)
        {
            int nthreads = threads_list[t];

            std::vector<double> x_blocks(n, 0.0);
            double T_blocks = run_solver(solve_jacobi_parallel_blocks,
                                         a.data(), b.data(), x_blocks.data(),
                                         n, MAX_ITER, TOL, nthreads);
            double S_blocks = (T_blocks > 0.0) ? T_serial / T_blocks : 0.0;

            std::vector<double> x_whole(n, 0.0);
            double T_whole = run_solver(solve_jacobi_parallel_whole,
                                        a.data(), b.data(), x_whole.data(),
                                        n, MAX_ITER, TOL, nthreads);
            double S_whole = (T_whole > 0.0) ? T_serial / T_whole : 0.0;

            std::cout << std::setw(8) << nthreads
                      << std::setw(15) << T_blocks
                      << std::setw(15) << S_blocks
                      << std::setw(15) << T_whole
                      << std::setw(15) << S_whole << "\n";

            used_threads.push_back(nthreads);
            T_blocks_vec.push_back(T_blocks);
            T_whole_vec.push_back(T_whole);
        }

        write_speedup_csv_jacobi(n, used_threads, T_serial, T_blocks_vec, T_whole_vec);
        std::cout << "Speedup data written to speedup_jacobi_" << n << ".csv\n";
    }
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    std::cout << "\n=== Jacobi solver experiments (Task 3) ===\n";
    run_experiments();

    return 0;
}


