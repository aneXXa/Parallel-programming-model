// Lab 2 Task 2: numerical integration with OpenMP
// - integrate: serial midpoint rule
// - integrate_omp: local sum + #pragma omp atomic
// - integrate_omp_atomic: plain atomic on each iteration
// - speedup analysis for 1,2,4,7,8,16,20,40 threads, nsteps = 40 000 000
// - system information (CPU, NUMA, memory, OS) and CSV output for plotting

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*f)(double), double left, double right, int n)
{
    double h = (right - left) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += f(left + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*f)(double), double left, double right, int n)
{
    double h = (right - left) / n;
    double sum = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0;

#pragma omp for nowait
        for (int i = 0; i < n; ++i)
        {
            local_sum += f(left + h * (i + 0.5));
        }

#pragma omp atomic
        sum += local_sum;
    }

    sum *= h;

    return sum;
}

double integrate_omp_atomic(double (*f)(double), double left, double right, int n)
{
    double h = (right - left) / n;
    double sum = 0.0;

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        double fx = f(left + h * (i + 0.5));
#pragma omp atomic
        sum += fx;
    }

    sum *= h;

    return sum;
}

double run_serial()
{
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
    (void)res; // uncomment printf below to check accuracy if needed
    // printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_parallel()
{
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps);
    t = cpuSecond() - t;
    (void)res;
    // printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_atomic()
{
    double t = cpuSecond();
    double res = integrate_omp_atomic(func, a, b, nsteps);
    t = cpuSecond() - t;
    (void)res;
    // printf("Result (atomic): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

void write_speedup_csv_integral(const std::vector<int> &threads,
                                const std::vector<double> &speedups_par,
                                const std::vector<double> &speedups_at)
{
    if (threads.empty() ||
        speedups_par.size() != threads.size() ||
        speedups_at.size() != threads.size())
        return;

    std::ostringstream name_suffix;
    name_suffix << nsteps;

    std::string csv_filename = "speedup_integral_" + name_suffix.str() + ".csv";

    std::ofstream csv_file(csv_filename.c_str());
    if (!csv_file)
    {
        std::cerr << "Cannot open file " << csv_filename << " for writing" << std::endl;
        return;
    }

    csv_file << "threads,S_par,S_at\n";
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        csv_file << threads[i] << "," << speedups_par[i] << "," << speedups_at[i] << "\n";
    }
    csv_file.close();
}

void run_experiment()
{
    const int threads_list[] = {1, 2, 4, 7, 8, 16, 20, 40};

    std::cout << std::fixed << std::setprecision(6);

    double T_serial = run_serial();

    std::cout << "T_serial = " << T_serial << " sec" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(8) << "n"
              << std::setw(15) << "T_par"
              << std::setw(15) << "S_par"
              << std::setw(15) << "T_at"
              << std::setw(15) << "S_at" << std::endl;

    std::vector<int> used_threads;
    std::vector<double> speedups_par;
    std::vector<double> speedups_at;

    for (std::size_t i = 0; i < sizeof(threads_list) / sizeof(threads_list[0]); ++i)
    {
        int nthreads = threads_list[i];

        omp_set_num_threads(nthreads);
        double T_par = run_parallel();

        omp_set_num_threads(nthreads);
        double T_at = run_atomic();

        double S_par = (T_par > 0.0) ? (T_serial / T_par) : 0.0;
        double S_at = (T_at > 0.0) ? (T_serial / T_at) : 0.0;

        std::cout << std::setw(8) << nthreads
                  << std::setw(15) << T_par
                  << std::setw(15) << S_par
                  << std::setw(15) << T_at
                  << std::setw(15) << S_at << std::endl;

        used_threads.push_back(nthreads);
        speedups_par.push_back(S_par);
        speedups_at.push_back(S_at);
    }

    write_speedup_csv_integral(used_threads, speedups_par, speedups_at);
}

void print_system_info()
{
    std::cout << std::endl
              << "=== CPU (lscpu) ===" << std::endl;
    std::cout << std::endl;
    std::system("lscpu");

    std::cout << std::endl
              << "=== Product name (cat /sys/devices/virtual/dmi/id/product_name) ===" << std::endl;
    std::cout << std::endl;
    std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo \"(no product_name in this environment)\"");

    std::cout << std::endl
              << "=== NUMA nodes (numactl --hardware) ===" << std::endl;
    std::cout << std::endl;
    std::system("numactl --hardware 2>/dev/null || echo \"(numactl not available or no NUMA info)\"");

    std::cout << std::endl
              << "=== Memory per node (from numactl/free) ===" << std::endl;
    std::cout << std::endl;
    std::system("free -h || echo \"(free not available)\"");

    std::cout << std::endl
              << "=== OS (cat /etc/os-release) ===" << std::endl;
    std::cout << std::endl;
    std::system("cat /etc/os-release");
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    print_system_info();

    std::cout << std::endl
              << "=== Integral speedup experiments (nsteps = 40 000 000) ===" << std::endl;

    run_experiment();

    std::cout << "Speedup data written to speedup_integral_" << nsteps << ".csv" << std::endl;

    return 0;
}

