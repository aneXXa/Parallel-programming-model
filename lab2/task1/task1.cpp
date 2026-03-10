/*
 * Lab 2 Task 1: Matrix-vector multiplication (DGEMV)
 * - Multithreaded version with parallel array initialization
 * - Scalability analysis for 1, 2, 4, 7, 8, 16, 20, 40 threads
 * - Matrix sizes: 20000x20000, 40000x40000 (double)
 * - System info: CPU, product name, NUMA, memory, OS
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

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

/*
 * matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
 * Serial version.
 */
void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

/*
 * matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n]
 * Parallel version — rows distributed across threads.
 */
void matrix_vector_product_omp(double *a, double *b, double *c, size_t m, size_t n)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = (int)(m / nthreads);
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (int)(m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (size_t j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

/*
 * Parallel initialization: fill matrix a[m][n] and vector b[n].
 * a[i][j] = i + j, b[j] = j.
 */
void init_arrays_parallel(double *a, double *b, size_t m, size_t n)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t rows_per_thread = m / (size_t)nthreads;
        size_t i_start = (size_t)tid * rows_per_thread;
        size_t i_end = (tid == nthreads - 1) ? m : (i_start + rows_per_thread);

        for (size_t i = i_start; i < i_end; i++)
        {
            for (size_t j = 0; j < n; j++)
                a[i * n + j] = (double)(i + j);
        }

        size_t vec_chunk = n / (size_t)nthreads;
        size_t j_start = (size_t)tid * vec_chunk;
        size_t j_end = (tid == nthreads - 1) ? n : (j_start + vec_chunk);
        for (size_t j = j_start; j < j_end; j++)
            b[j] = (double)j;
    }
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
        std::cerr << "Error allocate memory in run_serial for size "
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
        std::cerr << "Error allocate memory in run_parallel for size "
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

void run_experiments(std::ostream *report_file = nullptr)
{
    const size_t sizes[] = {20000, 40000};
    const int threads_list[] = {1, 2, 4, 7, 8, 16, 20, 40};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t num_threads = sizeof(threads_list) / sizeof(threads_list[0]);

    std::cout << std::fixed << std::setprecision(6);
    if (report_file)
        (*report_file) << std::fixed << std::setprecision(6);

    for (size_t sz_idx = 0; sz_idx < num_sizes; ++sz_idx)
    {
        size_t M = sizes[sz_idx];
        size_t N = sizes[sz_idx];

        std::cout << "\nMatrix size M = N = " << M << std::endl;

        double T_serial = run_serial(N, M);
        if (T_serial < 0.0)
        {
            std::cout << "Skip size " << M << " (not enough memory)" << std::endl;
            continue;
        }

        std::cout << "T_serial = " << T_serial << " sec\n" << std::endl;
        std::cout << std::setw(8) << "n"
                  << std::setw(15) << "T_n (sec)"
                  << std::setw(15) << "S_n=T_serial/Tn" << std::endl;

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
            double T_n = run_parallel(N, M, nthreads);
            if (T_n < 0.0)
            {
                std::cout << std::setw(8) << nthreads << std::setw(15) << "N/A" << std::setw(15) << "N/A" << std::endl;
                if (report_file)
                    *report_file << std::setw(8) << nthreads << std::setw(15) << "N/A" << std::setw(15) << "N/A" << "\n";
                continue;
            }
            double S_n = (T_n > 0.0) ? (T_serial / T_n) : 0.0;

            std::cout << std::setw(8) << nthreads << std::setw(15) << T_n << std::setw(15) << S_n << std::endl;
            if (report_file)
                *report_file << std::setw(8) << nthreads << std::setw(15) << T_n << std::setw(15) << S_n << "\n";

            used_threads.push_back(nthreads);
            used_speedups.push_back(S_n);
        }

        write_speedup_csv(M, used_threads, used_speedups);
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
    out << "\n=== Вывод о масштабируемости ===\n";
    out << "Ускорение S_n = T_serial / T_n. По результатам замеров: ускорение растёт с увеличением "
        << "числа потоков, но обычно меньше линейного (идеальное S_n = n). Причины: накладные расходы "
        << "на создание потоков и синхронизацию, неравномерная загрузка, ограниченная пропускная "
        << "способность памяти (задача memory-bound). Для матриц 20000x20000 и 40000x40000 масштабируемость "
        << "зависит от числа ядер и топологии NUMA. Для графиков программа сохраняет данные в CSV "
        << "(speedup_DGEMV_*.csv). PDF-графики можно построить скриптом lab2/plot_speedup.py "
        << "(если доступен Python+matplotlib) или любым табличным редактором с экспортом в PDF.\n";
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
        report_file << "=== System info: см. вывод программы в консоли (lscpu, numactl, /etc/os-release) ===\n";
        report_file << "\n=== Таблица масштабируемости (DGEMV, double, параллельная инициализация) ===\n";
    }

    std::cout << "\n=== DGEMV scalability tests (threads: 1,2,4,7,8,16,20,40; sizes: 20000, 40000) ===" << std::endl;

    run_experiments(report_file ? &report_file : nullptr);

    if (report_file)
    {
        write_report_conclusion(report_file);
        report_file.close();
        std::cout << "Report table and conclusion written to " << report_path
                  << " (speedup data: speedup_DGEMV_*.csv; generate PDFs via plot_speedup.py)\n";
    }

    return 0;
}
