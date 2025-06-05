#include <iostream>
#include <iomanip>
#include "price_generation.h"

int main() {
    double S0 = 100.0;
    double mu = 0.05;
    double sigma = 0.2;
    double T = 1.0;
    int steps = 10;
    int n_paths = 3;

    std::cout << "Simulating GBM (CPU) with S0=" << S0 << ", mu=" << mu << ", sigma=" << sigma
              << ", T=" << T << ", steps=" << steps << ", n_paths=" << n_paths << std::endl;

    Eigen::MatrixXd paths = simulateGBM_CPU(S0, mu, sigma, T, steps, n_paths);

    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < n_paths; ++i) {
        std::cout << "Path " << i << ": ";
        for (int t = 0; t <= steps; ++t) {
            std::cout << paths(i, t) << " ";
        }
        std::cout << std::endl;
    }

#ifdef __CUDACC__
    std::cout << "\nSimulating GBM (GPU)..." << std::endl;
    int blocks = 1, threads_per_block = n_paths;
    Eigen::MatrixXd gpu_paths = simulateGBM_GPU(S0, mu, sigma, T, steps, n_paths, blocks, threads_per_block);
    for (int i = 0; i < n_paths; ++i) {
        std::cout << "GPU Path " << i << ": ";
        for (int t = 0; t <= steps; ++t) {
            std::cout << gpu_paths(i, t) << " ";
        }
        std::cout << std::endl;
    }
#endif

    return 0;
}
