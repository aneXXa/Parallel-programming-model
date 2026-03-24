#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#include <omp.h>
#include <stdio.h>
#include <time.h>

static const int MAX_ITER = 1000;
static const double TOL = 1e-10;

static const std::size_t N_EXPERIMENT = 15000;
static const int OVERALL_RUNS_DEFAULT = 1;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

static double mean_drop_max(std::vector<double> samples, int drop_max)
{
    if (samples.empty())
        return -1.0;
    std::sort(samples.begin(), samples.end());
    if (drop_max > 0 && (int)samples.size() > drop_max)
        samples.resize(samples.size() - (size_t)drop_max);
    double sum = 0.0;
    for (double v : samples)
        sum += v;
    return sum / (double)samples.size();
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
    solver(a, b, x, n, max_iter, tol);
    t = cpuSecond() - t;
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

    csv << "threads,T_serial,T_blocks,S_blocks,E_blocks,T_whole,S_whole,E_whole\n";
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        double S_blocks = (T_blocks[i] > 0.0) ? (T_serial / T_blocks[i]) : 0.0;
        double S_whole = (T_whole[i] > 0.0) ? (T_serial / T_whole[i]) : 0.0;
        double E_blocks = (threads[i] > 0) ? (S_blocks / (double)threads[i]) : 0.0;
        double E_whole = (threads[i] > 0) ? (S_whole / (double)threads[i]) : 0.0;
        csv << threads[i] << ","
            << T_serial << ","
            << T_blocks[i] << "," << S_blocks << "," << E_blocks << ","
            << T_whole[i] << "," << S_whole << "," << E_whole << "\n";
    }
    csv.close();
}

// ---------- experiment driver ----------
void run_experiments(std::size_t n_input, int max_iter_input, bool verbose = true, bool write_csv = true)
{
    const std::size_t sizes[] = {n_input};
    const int threads_list[] = {1, 2, 4, 8, 16, 20, 40};
    const std::size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const std::size_t n_threads = sizeof(threads_list) / sizeof(threads_list[0]);

    const int repeats = 7;
    const int drop_max = 1;

    if (verbose)
        std::cout << std::fixed << std::setprecision(9);

    for (std::size_t s = 0; s < n_sizes; ++s)
    {
        std::size_t n = sizes[s];
        if (verbose)
            std::cout << "\n========== Jacobi: matrix size n = " << n << " ==========\n";

        std::vector<double> a, b, x_serial(n, 0.0);
        generate_system(a, b, n);

        std::vector<double> t_serial_samples;
        t_serial_samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            std::fill(x_serial.begin(), x_serial.end(), 0.0);
            t_serial_samples.push_back(run_solver(solve_jacobi_serial,
                                                  a.data(), b.data(), x_serial.data(),
                                                  n, max_iter_input, TOL, 1));
        }
        double T_serial = mean_drop_max(t_serial_samples, drop_max);
        if (verbose)
            std::cout << "T_serial = " << T_serial << " sec\n\n";

        if (verbose)
            std::cout << std::setw(8) << "threads"
                      << std::setw(15) << "T_blocks"
                      << std::setw(15) << "S_blocks"
                      << std::setw(15) << "E_blocks"
                      << std::setw(15) << "T_whole"
                      << std::setw(15) << "S_whole"
                      << std::setw(15) << "E_whole" << "\n";

        std::vector<int> used_threads;
        std::vector<double> T_blocks_vec;
        std::vector<double> T_whole_vec;

        for (std::size_t t = 0; t < n_threads; ++t)
        {
            int nthreads = threads_list[t];

            std::vector<double> t_blocks_samples;
            t_blocks_samples.reserve(repeats);
            for (int r = 0; r < repeats; ++r)
            {
                std::vector<double> x_blocks(n, 0.0);
                t_blocks_samples.push_back(run_solver(solve_jacobi_parallel_blocks,
                                                      a.data(), b.data(), x_blocks.data(),
                                                      n, max_iter_input, TOL, nthreads));
            }
            double T_blocks = mean_drop_max(t_blocks_samples, drop_max);
            double S_blocks = (T_blocks > 0.0) ? T_serial / T_blocks : 0.0;

            std::vector<double> t_whole_samples;
            t_whole_samples.reserve(repeats);
            for (int r = 0; r < repeats; ++r)
            {
                std::vector<double> x_whole(n, 0.0);
                t_whole_samples.push_back(run_solver(solve_jacobi_parallel_whole,
                                                     a.data(), b.data(), x_whole.data(),
                                                     n, max_iter_input, TOL, nthreads));
            }
            double T_whole = mean_drop_max(t_whole_samples, drop_max);
            double S_whole = (T_whole > 0.0) ? T_serial / T_whole : 0.0;
            double E_blocks = (nthreads > 0) ? S_blocks / (double)nthreads : 0.0;
            double E_whole = (nthreads > 0) ? S_whole / (double)nthreads : 0.0;

            if (verbose)
                std::cout << std::setw(8) << nthreads
                          << std::setw(15) << T_blocks
                          << std::setw(15) << S_blocks
                          << std::setw(15) << E_blocks
                          << std::setw(15) << T_whole
                          << std::setw(15) << S_whole
                          << std::setw(15) << E_whole << "\n";

            used_threads.push_back(nthreads);
            T_blocks_vec.push_back(T_blocks);
            T_whole_vec.push_back(T_whole);
        }

        if (write_csv)
            write_speedup_csv_jacobi(n, used_threads, T_serial, T_blocks_vec, T_whole_vec);
        if (verbose)
            std::cout << "Speedup data written to speedup_jacobi_" << n << ".csv\n";
    }
}

int main(int argc, char **argv)
{
    std::size_t n_value = N_EXPERIMENT;
    int max_iter_value = MAX_ITER;
    int overall_repeats = OVERALL_RUNS_DEFAULT;

    if (argc > 1)
        n_value = (std::size_t)std::strtoull(argv[1], nullptr, 10);
    if (argc > 2)
        max_iter_value = std::atoi(argv[2]);
    if (argc > 3)
        overall_repeats = std::atoi(argv[3]);
    if (overall_repeats < 1)
        overall_repeats = 1;

    std::cout << "Task3 config: N=" << n_value
              << ", MAX_ITER=" << max_iter_value
              << ", overall full runs=" << overall_repeats << std::endl;

    const int overall_drop_max = 1;
    std::vector<double> total_samples;
    total_samples.reserve(overall_repeats);
    for (int r = 0; r < overall_repeats; ++r)
    {
        std::cout << "Running full experiment pass " << (r + 1)
                  << "/" << overall_repeats << "..." << std::endl;
        double t0 = cpuSecond();
        run_experiments(n_value, max_iter_value, false, false);
        total_samples.push_back(cpuSecond() - t0);
    }
    double T_total_mean = mean_drop_max(total_samples, overall_drop_max);

    std::cout << "\nMean total runtime over " << overall_repeats
              << " full runs (drop max " << overall_drop_max << "): "
              << T_total_mean << " sec\n";

    std::cout << "\n=== Jacobi solver experiments (Task 3) ===\n";
    run_experiments(n_value, max_iter_value, true, true);

    return 0;
}


