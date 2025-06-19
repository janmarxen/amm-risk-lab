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

    // // Test final_only option (CPU)
    // Eigen::MatrixXd final_prices = simulateGBM(S0, mu, sigma, T, steps, n_paths, false, 0, 0, true);
    // std::cout << std::fixed << std::setprecision(4);
    // std::cout << "Final prices (CPU, final_only):" << std::endl;
    // for (int i = 0; i < n_paths; ++i) {
    //     std::cout << "Path " << i << ": " << final_prices(i, 0) << std::endl;
    // }

    // Test simulateGBMSeries (high-level API)
    std::string start_datetime = "2023-01-01T00:00:00";
    std::string end_datetime = "2023-01-01T10:00:00";
    int n_paths_series = 3;
    std::cout << "\nTesting simulateGBMSeries from " << start_datetime << " to " << end_datetime << " with n_paths=" << n_paths_series << std::endl;
    PriceSeries series = simulateGBMSeries(
        S0, mu, sigma, start_datetime, end_datetime, n_paths_series, false, 0, 0, false
    );
    std::cout << "Datetime\t\t";
    for (int t = 0; t < series.datetimes.size(); ++t) {
        std::cout << series.datetimes[t] << (t + 1 < series.datetimes.size() ? ", " : "\n");
    }
    for (int i = 0; i < n_paths_series; ++i) {
        std::cout << "Path " << i << ": ";
        for (int t = 0; t < series.prices.cols(); ++t) {
            std::cout << series.prices(i, t) << (t + 1 < series.prices.cols() ? ", " : "\n");
        }
    }

    // // Test simulateGBMSeries with final_only
    // PriceSeries series_final = simulateGBMSeries(
    //     S0, mu, sigma, start_datetime, end_datetime, n_paths_series, false, 0, 0, true
    // );
    // std::cout << "\nTesting simulateGBMSeries (final_only):" << std::endl;
    // for (int i = 0; i < n_paths_series; ++i) {
    //     std::cout << "Path " << i << " final price at " << series_final.datetimes[0] << ": " << series_final.prices(i, 0) << std::endl;
    // }

    return 0;
}
