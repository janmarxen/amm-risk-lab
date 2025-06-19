#pragma once

#include "lp_simulation.h"
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @brief Struct to hold Monte Carlo statistics for a field.
 */
struct MonteCarloStats {
    Eigen::VectorXd mean;
    Eigen::VectorXd std;
    std::vector<std::pair<double, Eigen::VectorXd>> quantiles; // pair<quantile, values>
};

Eigen::VectorXd monte_carlo_mean(const Eigen::MatrixXd& mat);
Eigen::VectorXd monte_carlo_std(const Eigen::MatrixXd& mat);
Eigen::VectorXd monte_carlo_quantile(const Eigen::MatrixXd& mat, double quantile);

MonteCarloStats monte_carlo_stats(
    const LPSimulationResults& results,
    const std::string& field,
    const std::vector<double>& quantiles
);

void print_monte_carlo_stats(const MonteCarloStats& stats, const std::vector<std::string>& datetimes, const std::string& label);

/**
 * @brief Struct to hold summary statistics for all relevant fields.
 */
struct MonteCarloSummary {
    std::vector<std::string> fields;
    std::vector<MonteCarloStats> stats;
};

MonteCarloSummary summarize_all_stats(
    const LPSimulationResults& results,
    const std::vector<std::string>& datetimes,
    const std::vector<double>& quantiles = {0.05, 0.5, 0.95}
);

void print_monte_carlo_summary(
    const MonteCarloSummary& summary,
    const std::vector<std::string>& datetimes
);
