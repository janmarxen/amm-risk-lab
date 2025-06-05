#include <iostream>
#include <iomanip>
#include "price_generation.h"

int main() {
    double S0 = 100.0;
    double mu = 0.0;
    double sigma = 0.001; 
    double T = 1.0;
    int steps = 24;
    int n_paths = 100;

    std::cout << "Simulating GBM (CPU) with S0=" << S0 << ", mu=" << mu << ", sigma=" << sigma
              << ", T=" << T << ", steps=" << steps << ", n_paths=" << n_paths << std::endl;

    // Eigen::MatrixXd paths = simulateGBM_CPU(S0, mu, sigma, T, steps, n_paths);

    // std::cout << std::fixed << std::setprecision(4);
    // for (int i = 0; i < n_paths; ++i) {
    //     std::cout << "Path " << i << ": ";
    //     for (int t = 0; t <= steps; ++t) {
    //         std::cout << paths(i, t) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Test final_only option (CPU)
    Eigen::MatrixXd final_prices = simulateGBM(S0, mu, sigma, T, steps, n_paths, false, 0, 0, true);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Final prices (CPU, final_only):" << std::endl;
    for (int i = 0; i < n_paths; ++i) {
        std::cout << "Path " << i << ": " << final_prices(i, 0) << std::endl;
    }

#ifdef __CUDACC__
    // std::cout << "\nSimulating GBM (GPU)..." << std::endl;
    // int blocks = 1, threads_per_block = n_paths;
    // Eigen::MatrixXd gpu_paths = simulateGBM_GPU(S0, mu, sigma, T, steps, n_paths, blocks, threads_per_block);
    // for (int i = 0; i < n_paths; ++i) {
    //     std::cout << "GPU Path " << i << ": ";
    //     for (int t = 0; t <= steps; ++t) {
    //         std::cout << gpu_paths(i, t) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Test final_only option (GPU)
    int blocks = 1, threads_per_block = n_paths;
    Eigen::MatrixXd gpu_final_prices = simulateGBM(S0, mu, sigma, T, steps, n_paths, true, blocks, threads_per_block, true);
    std::cout << "Final prices (GPU, final_only):" << std::endl;
    for (int i = 0; i < n_paths; ++i) {
        std::cout << "GPU Path " << i << ": " << gpu_final_prices(i, 0) << std::endl;
    }
#endif

    return 0;
}
