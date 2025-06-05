#pragma once

#include <Eigen/Dense>
#include "common.h"

// Structs for simulation parameters
struct PriceSimulationParams {
    float mu;
    double sigma;
};

/**
 * @brief Simulate Geometric Brownian Motion (GBM) price paths on CPU.
 * 
 * @param S0 Initial price.
 * @param mu Drift parameter.
 * @param sigma Volatility parameter.
 * @param T Total time horizon (in hours).
 * @param steps Number of time steps.
 * @param n_paths Number of simulation paths.
 * @return Eigen::MatrixXd Matrix of simulated prices (n_paths x (steps+1)).
 */
Eigen::MatrixXd simulateGBM_CPU(double S0, double mu, double sigma, double T, int steps, int n_paths);

/**
 * @brief Simulate GBM price paths on CPU, returning only the final prices.
 * 
 * @param S0 Initial price.
 * @param mu Drift parameter.
 * @param sigma Volatility parameter.
 * @param T Total time horizon (in hours).
 * @param steps Number of time steps.
 * @param n_paths Number of simulation paths.
 * @return Eigen::MatrixXd Matrix of final prices (n_paths x 1).
 */
Eigen::MatrixXd simulateGBM_CPU_final_only(double S0, double mu, double sigma, double T, int steps, int n_paths);

#ifdef __CUDACC__
/**
 * @brief Simulate GBM price paths on GPU.
 * 
 * @param S0 Initial price.
 * @param mu Drift parameter.
 * @param sigma Volatility parameter.
 * @param T Total time horizon (in hours).
 * @param steps Number of time steps.
 * @param n_paths Number of simulation paths.
 * @param blocks Number of CUDA blocks.
 * @param threads_per_block Number of CUDA threads per block.
 * @return Eigen::MatrixXd Matrix of simulated prices (n_paths x (steps+1)).
 */
Eigen::MatrixXd simulateGBM_GPU(double S0, double mu, double sigma, double T, int steps, int n_paths, int blocks, int threads_per_block);

/**
 * @brief Simulate GBM price paths on GPU, returning only the final prices.
 * 
 * @param S0 Initial price.
 * @param mu Drift parameter.
 * @param sigma Volatility parameter.
 * @param T Total time horizon (in hours).
 * @param steps Number of time steps.
 * @param n_paths Number of simulation paths.
 * @param blocks Number of CUDA blocks.
 * @param threads_per_block Number of CUDA threads per block.
 * @return Eigen::MatrixXd Matrix of final prices (n_paths x 1).
 */
Eigen::MatrixXd simulateGBM_GPU_final_only(double S0, double mu, double sigma, double T, int steps, int n_paths, int blocks, int threads_per_block);
#endif

/**
 * @brief Dispatcher for GBM simulation (CPU or GPU). Optionally returns only final prices.
 * 
 * @param S0 Initial price.
 * @param mu Drift parameter.
 * @param sigma Volatility parameter.
 * @param T Total time horizon (in hours).
 * @param steps Number of time steps.
 * @param n_paths Number of simulation paths.
 * @param use_gpu Whether to use GPU (true) or CPU (false).
 * @param blocks Number of CUDA blocks (GPU only).
 * @param threads_per_block Number of CUDA threads per block (GPU only).
 * @param final_only If true, returns only final prices (n_paths x 1).
 * @return Eigen::MatrixXd Matrix of simulated prices.
 */
Eigen::MatrixXd simulateGBM(double S0, double mu, double sigma, double T, int steps, int n_paths, bool use_gpu = false, int blocks = 0, int threads_per_block = 0, bool final_only = false);

/**
 * @brief High-level GBM simulation interface. User provides start/end datetime (ISO 8601), hourly steps are generated.
 * 
 * @param S0 Initial price.
 * @param mu Drift parameter.
 * @param sigma Volatility parameter.
 * @param start_datetime Start datetime (ISO 8601, e.g. "2023-01-01T00:00:00").
 * @param end_datetime End datetime (ISO 8601, e.g. "2023-01-02T00:00:00").
 * @param n_paths Number of simulation paths.
 * @param use_gpu Whether to use GPU (true) or CPU (false).
 * @param blocks Number of CUDA blocks (GPU only).
 * @param threads_per_block Number of CUDA threads per block (GPU only).
 * @param final_only If true, returns only final prices (n_paths x 1).
 * @return PriceSeries Struct containing vector of datetimes and matrix of prices.
 */
PriceSeries simulateGBMSeries(
    double S0,
    double mu,
    double sigma,
    const std::string& start_datetime,
    const std::string& end_datetime,
    int n_paths,
    bool use_gpu = false,
    int blocks = 0,
    int threads_per_block = 0,
    bool final_only = false
);
