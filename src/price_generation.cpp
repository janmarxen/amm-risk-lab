#include "price_generation.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// CPU version of GBM simulation
MatrixXd simulateGBM_CPU(double S0, double mu, double sigma, double T, int steps, int n_paths) {
    double dt = T / steps;
    std::mt19937_64 rng(42); // fixed seed for reproducibility
    std::normal_distribution<double> norm(0.0, 1.0);

    MatrixXd paths(n_paths, steps + 1);
    paths.col(0).setConstant(S0);

    for (int i = 0; i < n_paths; ++i) {
        for (int t = 1; t <= steps; ++t) {
            double Z = norm(rng);
            paths(i, t) = paths(i, t - 1) * std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        }
    }
    return paths;
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

MatrixXd simulateGBM_GPU(double S0, double mu, double sigma, double T, int steps, int n_paths, int blocks, int threads_per_block) {
    double dt = T / steps;
    size_t total_size = n_paths * (steps + 1);
    double* d_paths;
    cudaMalloc(&d_paths, total_size * sizeof(double));

    simulateGBM_CUDA<<<blocks, threads_per_block>>>(d_paths, S0, mu, sigma, dt, steps, n_paths, 42);
    cudaDeviceSynchronize();

    std::vector<double> h_paths(total_size);
    cudaMemcpy(h_paths.data(), d_paths, total_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_paths);

    // MatrixXd result(n_paths, steps + 1);
    // for (int i = 0; i < n_paths; ++i)
    //     for (int t = 0; t <= steps; ++t)
    //         result(i, t) = h_paths[i * (steps + 1) + t];
    MatrixXd result = Eigen::Map<MatrixXd>(h_paths.data(), n_paths, steps + 1).eval();

    return result;
}
#endif

// Dispatcher function
MatrixXd simulateGBM(double S0, double mu, double sigma, double T, int steps, int n_paths, bool use_gpu = false, int blocks = 0, int threads_per_block = 0) {
#ifdef __CUDACC__
    if (use_gpu) {
        if (blocks == 0 || threads_per_block == 0) {
            std::cerr << "[Error] CUDA mode requires blocks and threads_per_block to be specified." << std::endl;
            exit(1);
        }
        return simulateGBM_GPU(S0, mu, sigma, T, steps, n_paths, blocks, threads_per_block);
    } else {
        return simulateGBM_CPU(S0, mu, sigma, T, steps, n_paths);
    }
#else
    if (use_gpu) {
        std::cerr << "[Error] CUDA requested but this build is not compiled with CUDA support." << std::endl;
        exit(1);
    }
    return simulateGBM_CPU(S0, mu, sigma, T, steps, n_paths);
#endif
}







