#include "price_generation.h"
#include "common.h"
#include "utils/time_utils.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

Eigen::MatrixXd simulateGBM_CPU(double S0, double mu, double sigma, double T, int steps, int n_paths) {
    double dt = T / steps;
    std::mt19937_64 rng(42); // fixed seed for reproducibility
    std::normal_distribution<double> norm(0.0, 1.0);

    Eigen::MatrixXd paths(n_paths, steps + 1);
    paths.col(0).setConstant(S0);

    for (int i = 0; i < n_paths; ++i) {
        for (int t = 1; t <= steps; ++t) {
            double Z = norm(rng);
            paths(i, t) = paths(i, t - 1) * std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        }
    }
    return paths;
}

Eigen::MatrixXd simulateGBM_CPU_final_only(double S0, double mu, double sigma, double T, int steps, int n_paths) {
    double dt = T / steps;
    std::mt19937_64 rng(42); // fixed seed for reproducibility
    std::normal_distribution<double> norm(0.0, 1.0);

    Eigen::MatrixXd finals(n_paths, 1);
    for (int i = 0; i < n_paths; ++i) {
        double price = S0;
        for (int t = 1; t <= steps; ++t) {
            double Z = norm(rng);
            price *= std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        }
        finals(i, 0) = price;
    }
    return finals;
}

#ifdef __CUDACC__
__global__ void simulateGBM_CUDA(double* d_paths, double S0, double mu, double sigma, double dt, int steps, int n_paths, unsigned long seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_paths) return;

    curandStateXORWOW_t state;
    curand_init(seed, idx, 0, &state);

    d_paths[idx * (steps + 1)] = S0;
    for (int t = 1; t <= steps; ++t) {
        double Z = curand_normal(&state);
        double prev = d_paths[idx * (steps + 1) + t - 1];
        d_paths[idx * (steps + 1) + t] = prev * exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
    }
}

Eigen::MatrixXd simulateGBM_GPU(double S0, double mu, double sigma, double T, int steps, int n_paths, int blocks, int threads_per_block) {
    double dt = T / steps;
    size_t total_size = n_paths * (steps + 1);
    double* d_paths;
    cudaMalloc(&d_paths, total_size * sizeof(double));

    simulateGBM_CUDA<<<blocks, threads_per_block>>>(d_paths, S0, mu, sigma, dt, steps, n_paths, 42);
    cudaDeviceSynchronize();

    std::vector<double> h_paths(total_size);
    cudaMemcpy(h_paths.data(), d_paths, total_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_paths);

    Eigen::MatrixXd result = Eigen::Map<Eigen::MatrixXd>(h_paths.data(), n_paths, steps + 1).eval();

    return result;
}

// GPU version with option to return only final prices as MatrixXd (n_paths x 1)
Eigen::MatrixXd simulateGBM_GPU_final_only(double S0, double mu, double sigma, double T, int steps, int n_paths, int blocks, int threads_per_block) {
    double dt = T / steps;
    size_t total_size = n_paths * (steps + 1);
    double* d_paths;
    cudaMalloc(&d_paths, total_size * sizeof(double));

    simulateGBM_CUDA<<<blocks, threads_per_block>>>(d_paths, S0, mu, sigma, dt, steps, n_paths, 42);
    cudaDeviceSynchronize();

    std::vector<double> h_paths(total_size);
    cudaMemcpy(h_paths.data(), d_paths, total_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_paths);

    Eigen::MatrixXd finals(n_paths, 1);
    for (int i = 0; i < n_paths; ++i) {
        finals(i, 0) = h_paths[i * (steps + 1) + steps];
    }
    return finals;
}
#endif

Eigen::MatrixXd simulateGBM(double S0, double mu, double sigma, double T, int steps, int n_paths, bool use_gpu, int blocks, int threads_per_block, bool final_only) {
#ifdef __CUDACC__
    if (use_gpu) {
        if (blocks == 0 || threads_per_block == 0) {
            std::cerr << "[Error] CUDA mode requires blocks and threads_per_block to be specified." << std::endl;
            exit(1);
        }
        if (final_only) {
            return simulateGBM_GPU_final_only(S0, mu, sigma, T, steps, n_paths, blocks, threads_per_block);
        }
        return simulateGBM_GPU(S0, mu, sigma, T, steps, n_paths, blocks, threads_per_block);
    } else {
        if (final_only) {
            return simulateGBM_CPU_final_only(S0, mu, sigma, T, steps, n_paths);
        }
        return simulateGBM_CPU(S0, mu, sigma, T, steps, n_paths);
    }
#else
    if (use_gpu) {
        std::cerr << "[Error] CUDA requested but this build is not compiled with CUDA support." << std::endl;
        exit(1);
    }
    if (final_only) {
        return simulateGBM_CPU_final_only(S0, mu, sigma, T, steps, n_paths);
    }
    return simulateGBM_CPU(S0, mu, sigma, T, steps, n_paths);
#endif
}


PriceSeries simulateGBMSeries(
    double S0,
    double mu,
    double sigma,
    const std::string& start_datetime,
    const std::string& end_datetime,
    int n_paths,
    bool use_gpu,
    int blocks,
    int threads_per_block,
    bool final_only
) {
    std::tm tm_start = parse_iso_datetime(start_datetime);
    std::tm tm_end = parse_iso_datetime(end_datetime);

    std::time_t t_start = std::mktime(&tm_start);
    std::time_t t_end = std::mktime(&tm_end);

    if (t_end <= t_start) {
        throw std::runtime_error("end_datetime must be after start_datetime");
    }

    int steps = static_cast<int>(std::difftime(t_end, t_start) / 3600); // number of hours
    double T = steps; // T in hours

    // Generate datetime vector
    std::vector<std::string> datetimes;
    std::tm tm_current = tm_start;
    for (int i = 0; i <= steps; ++i) {
        datetimes.push_back(format_iso_datetime(tm_current));
        std::time_t t = std::mktime(&tm_current) + 3600;
        tm_current = *std::localtime(&t);
    }
    if (final_only) {
        datetimes = {format_iso_datetime(tm_end)};
    }

    Eigen::MatrixXd prices = simulateGBM(S0, mu, sigma, T, steps, n_paths, use_gpu, blocks, threads_per_block, final_only);

    PriceSeries result;
    result.datetimes = datetimes;
    result.prices = prices;
    return result;
}







