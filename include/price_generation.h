#pragma once

#include <Eigen/Dense>

// Structs for simulation parameters
struct PriceSimulationParams {
    float mu;
    double sigma;
};

// CPU version of GBM simulation
Eigen::MatrixXd simulateGBM_CPU(double S0, double mu, double sigma, double T, int steps, int n_paths);

#ifdef __CUDACC__
// GPU version of GBM simulation
Eigen::MatrixXd simulateGBM_GPU(double S0, double mu, double sigma, double T, int steps, int n_paths, int blocks, int threads_per_block);
#endif

// Dispatcher function (CPU or GPU based on use_gpu flag)
Eigen::MatrixXd simulateGBM(double S0, double mu, double sigma, double T, int steps, int n_paths, bool use_gpu = false, int blocks = 0, int threads_per_block = 0);
