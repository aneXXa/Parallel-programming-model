// Lab 2 Task 1: DGEMV with OpenMP
// - Serial and parallel matrix–vector product
// - Parallel initialization of A and b
// - Scalability study for 1,2,4,7,8,16,20,40 threads and sizes 20000/40000
// - System information: CPU, product name, NUMA, memory, OS

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

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

// Serial matrix–vector product: c[m] = A[m][n] * b[n]
void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

// Parallel matrix–vector product: rows are distributed across threads
void matrix_vector_product_omp(double *a, double *b, double *c, size_t m, size_t n)
{
#pragma omp parallel for
    for (size_t i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

// Parallel initialization: A[i,j] = i + j, b[j] = j
void init_arrays_parallel(double *a, double *b, size_t m, size_t n)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = (double)(i + j);
    }

#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; j++)
        b[j] = (double)j;
}

double run_serial(size_t n, size_t m)
{
    double *a = (double *)malloc(sizeof(*a) * m * n);
    double *b = (double *)malloc(sizeof(*b) * n);
    double *c = (double *)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        std::cerr << "Error allocating memory in run_serial for size "
                  << "M=" << m << ", N=" << n << std::endl;
        return -1.0;
    }

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = (double)(i + j);
    }
    for (size_t j = 0; j < n; j++)
        b[j] = (double)j;

    double t = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t = cpuSecond() - t;

    free(a);
    free(b);
    free(c);
    return t;
}

double run_parallel(size_t n, size_t m, int nthreads)
{
    double *a = (double *)malloc(sizeof(*a) * m * n);
    double *b = (double *)malloc(sizeof(*b) * n);
    double *c = (double *)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        std::cerr << "Error allocating memory in run_parallel for size "
                  << "M=" << m << ", N=" << n
                  << ", threads=" << nthreads << std::endl;
        return -1.0;
    }

    omp_set_num_threads(nthreads);
    init_arrays_parallel(a, b, m, n);

    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n);
    t = cpuSecond() - t;

    free(a);
    free(b);
    free(c);
    return t;
}

void write_speedup_csv(size_t size, const std::vector<int> &threads, const std::vector<double> &speedups)
{
    if (threads.empty() || speedups.empty() || threads.size() != speedups.size())
        return;

    std::ostringstream name_suffix;
    name_suffix << size;

    std::string csv_filename = "speedup_DGEMV_" + name_suffix.str() + ".csv";

    std::ofstream csv_file(csv_filename.c_str());
    if (!csv_file)
    {
        std::cerr << "Cannot open file " << csv_filename << " for writing" << std::endl;
        return;
    }
    csv_file << "threads,speedup\n";
    for (std::size_t i = 0; i < threads.size(); ++i)
        csv_file << threads[i] << "," << speedups[i] << "\n";
    csv_file.close();
}

void run_experiments(std::ostream *report_file = nullptr, bool verbose = true)
{
    const size_t sizes[] = {20000, 40000};
    const int threads_list[] = {1, 2, 4, 7, 8, 16, 20, 40};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t num_threads = sizeof(threads_list) / sizeof(threads_list[0]);

    const int repeats = 7;
    const int drop_max = 1;

    if (verbose)
        std::cout << std::fixed << std::setprecision(9);
    if (report_file)
        (*report_file) << std::fixed << std::setprecision(9);

    for (size_t sz_idx = 0; sz_idx < num_sizes; ++sz_idx)
    {
        size_t M = sizes[sz_idx];
        size_t N = sizes[sz_idx];

        if (verbose)
            std::cout << "\nMatrix size M = N = " << M << std::endl;

        double *a = (double *)malloc(sizeof(double) * M * N);
        double *b = (double *)malloc(sizeof(double) * N);
        double *c = (double *)malloc(sizeof(double) * M);

        if (!a || !b || !c)
        {
            free(a);
            free(b);
            free(c);
            if (verbose)
                std::cout << "Skip size " << M << " (not enough memory)" << std::endl;
            continue;
        }

        omp_set_num_threads(1);
        init_arrays_parallel(a, b, M, N);

        std::vector<double> t_serial_samples;
        t_serial_samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            double t0 = cpuSecond();
            matrix_vector_product(a, b, c, M, N);
            t_serial_samples.push_back(cpuSecond() - t0);
        }
        double T_serial = mean_drop_max(t_serial_samples, drop_max);

        if (verbose)
        {
            std::cout << "T_serial = " << T_serial << " sec\n" << std::endl;
            std::cout << std::setw(8) << "n"
                      << std::setw(15) << "T_n (sec)"
                      << std::setw(15) << "S_n=T_serial/Tn" << std::endl;
        }

        if (report_file)
        {
            *report_file << "\n--- Matrix size M = N = " << M << " ---\n";
            *report_file << "T_serial = " << T_serial << " sec\n";
            *report_file << std::setw(8) << "threads" << std::setw(15) << "T_n (sec)" << std::setw(15) << "S_n" << "\n";
        }

        std::vector<int> used_threads;
        std::vector<double> used_speedups;

        for (size_t i = 0; i < num_threads; ++i)
        {
            int nthreads = threads_list[i];
            omp_set_num_threads(nthreads);

            std::vector<double> t_samples;
            t_samples.reserve(repeats);
            for (int r = 0; r < repeats; ++r)
            {
                double t0 = cpuSecond();
                matrix_vector_product_omp(a, b, c, M, N);
                t_samples.push_back(cpuSecond() - t0);
            }
            double T_n = mean_drop_max(t_samples, drop_max);

            double S_n = (T_n > 0.0) ? (T_serial / T_n) : 0.0;

            if (verbose)
                std::cout << std::setw(8) << nthreads << std::setw(15) << T_n << std::setw(15) << S_n << std::endl;
            if (report_file)
                *report_file << std::setw(8) << nthreads << std::setw(15) << T_n << std::setw(15) << S_n << "\n";

            used_threads.push_back(nthreads);
            used_speedups.push_back(S_n);
        }

        write_speedup_csv(M, used_threads, used_speedups);

        free(a);
        free(b);
        free(c);
    }
}

void print_system_info(std::ostream &out)
{
    out << "=== CPU (lscpu) ===" << std::endl;
    std::system("lscpu 2>nul || lscpu 2>/dev/null || echo \"(lscpu not available on this system)\"");

    out << "\n=== Product name (server) ===" << std::endl;
    std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || type %SYSTEMROOT%\\System32\\drivers\\etc\\hostname 2>nul || echo \"(product_name not available)\"");

    out << "\n=== NUMA nodes (numactl --hardware) ===" << std::endl;
    std::system("numactl --hardware 2>/dev/null || numactl --hardware 2>nul || echo \"(numactl not available)\"");

    out << "\n=== Memory per node / total ===" << std::endl;
    std::system("free -h 2>/dev/null || wmic OS get FreePhysicalMemory,TotalVisibleMemorySize 2>nul || echo \"(memory info not available)\"");

    out << "\n=== OS (cat /etc/os-release) ===" << std::endl;
    std::system("cat /etc/os-release 2>/dev/null || ver 2>nul || echo \"(os-release not available)\"");
}

void write_report_conclusion(std::ostream &out)
{
    out << "\n=== Scalability discussion ===\n";
    out << "Speedup S_n = T_serial / T_n grows with the number of threads but is typically sub-linear "
        << "(ideal S_n = n). Reasons include thread creation and synchronization overheads, load imbalance, "
        << "and limited memory bandwidth (DGEMV is memory-bound). For matrices 20000x20000 and 40000x40000 "
        << "scalability also depends on the number of physical cores and NUMA topology. The program saves "
        << "speedup data to CSV files (speedup_DGEMV_*.csv); PDF plots can be generated with plot_speedup.py "
        << "or any spreadsheet tool that can export charts to PDF.\n";
}

int main(int argc, char *argv[])
{
    std::cout << "=== System information ===" << std::endl;
    print_system_info(std::cout);

    std::string report_path = "task1_report.txt";
    if (argc > 1)
        report_path = argv[1];

    std::ofstream report_file(report_path);
    if (report_file)
    {
        report_file << "=== System info: see console output (lscpu, numactl, /etc/os-release) ===\n";
        report_file << "\n=== Scalability table (DGEMV, double, parallel initialization) ===\n";
    }

    const int overall_repeats = 3;
    const int overall_drop_max = 1;
    std::vector<double> total_samples;
    total_samples.reserve(overall_repeats);
    for (int r = 0; r < overall_repeats; ++r)
    {
        double t0 = cpuSecond();
        run_experiments(nullptr, false);
        total_samples.push_back(cpuSecond() - t0);
    }
    double T_total_mean = mean_drop_max(total_samples, overall_drop_max);
    std::cout << "\nMean total runtime over " << overall_repeats
              << " full runs (drop max " << overall_drop_max << "): "
              << T_total_mean << " sec\n";

    std::cout << "\n=== DGEMV scalability tests (threads: 1,2,4,7,8,16,20,40; sizes: 20000, 40000) ===" << std::endl;
    run_experiments(report_file ? &report_file : nullptr, true);

    if (report_file)
    {
        write_report_conclusion(report_file);
        report_file.close();
        std::cout << "Report table and conclusion written to " << report_path
                  << " (speedup data: speedup_DGEMV_*.csv; generate PDFs via plot_speedup.py)\n";
    }

    return 0;
}
