#include "monte_carlo.h"
#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

static const Eigen::MatrixXd& get_field(const LPSimulationResults& results, const std::string& field) {
    if (field == "user_liquidity") return results.user_liquidity;
    if (field == "token0_amounts") return results.token0_amounts;
    if (field == "token1_amounts") return results.token1_amounts;
    if (field == "feesUSD_acc") return results.feesUSD_acc;
    if (field == "il_usd") return results.il_usd;
    throw std::invalid_argument("Unknown field: " + field);
}

Eigen::VectorXd monte_carlo_mean(const Eigen::MatrixXd& mat) {
    return mat.colwise().mean();
}

Eigen::VectorXd monte_carlo_std(const Eigen::MatrixXd& mat) {
    Eigen::ArrayXXd centered = mat.rowwise() - mat.colwise().mean();
    Eigen::ArrayXd var = (centered.square().colwise().sum()) / (mat.rows() - 1);
    return var.sqrt().matrix();
}

Eigen::VectorXd monte_carlo_quantile(const Eigen::MatrixXd& mat, double quantile) {
    int n_paths = mat.rows();
    int n_points = mat.cols();
    Eigen::VectorXd quant(n_points);
    Eigen::MatrixXd sorted = mat;
    for (int j = 0; j < n_points; ++j) {
        Eigen::VectorXd col = sorted.col(j);
        std::vector<double> col_vec(col.data(), col.data() + col.size());
        std::sort(col_vec.begin(), col_vec.end());
        double pos = quantile * (n_paths - 1);
        int idx_below = static_cast<int>(std::floor(pos));
        int idx_above = static_cast<int>(std::ceil(pos));
        if (idx_below == idx_above) {
            quant(j) = col_vec[idx_below];
        } else {
            double weight_above = pos - idx_below;
            quant(j) = col_vec[idx_below] * (1.0 - weight_above) + col_vec[idx_above] * weight_above;
        }
    }
    return quant;
}

MonteCarloStats monte_carlo_stats(
    const LPSimulationResults& results,
    const std::string& field,
    const std::vector<double>& quantiles
) {
    const Eigen::MatrixXd& mat = get_field(results, field);
    MonteCarloStats stats;
    stats.mean = monte_carlo_mean(mat);
    stats.std = monte_carlo_std(mat);
    for (double q : quantiles) {
        stats.quantiles.emplace_back(q, monte_carlo_quantile(mat, q));
    }
    return stats;
}

void print_monte_carlo_stats(const MonteCarloStats& stats, const std::vector<std::string>& datetimes, const std::string& label) {
    std::cout << "Monte Carlo statistics for " << label << ":\n";
    int n = static_cast<int>(datetimes.size());
    int width = 20;

    // Print header row
    std::cout << std::left << std::setw(width) << "Datetime"
              << std::setw(width) << "Mean"
              << std::setw(width) << "Std";
    for (const auto& q : stats.quantiles) {
        std::ostringstream qlabel;
        qlabel << "Q" << std::fixed << std::setprecision(2) << q.first;
        std::cout << std::setw(width) << qlabel.str();
    }
    std::cout << "\n";

    // Print each row: datetime, mean, std, quantiles
    for (int i = 0; i < n; ++i) {
        std::cout << std::left << std::setw(width) << datetimes[i]
                  << std::setw(width) << stats.mean(i)
                  << std::setw(width) << stats.std(i);
        for (const auto& q : stats.quantiles) {
            std::cout << std::setw(width) << q.second(i);
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

MonteCarloSummary summarize_all_stats(
    const LPSimulationResults& results,
    const std::vector<std::string>& datetimes,
    const std::vector<double>& quantiles
) {
    std::vector<std::string> fields = {
        "user_liquidity", "token0_amounts", "token1_amounts", "feesUSD_acc", "il_usd"
    };
    MonteCarloSummary summary;
    summary.fields = fields;
    for (const auto& field : fields) {
        MonteCarloStats stats = monte_carlo_stats(results, field, quantiles);
        summary.stats.push_back(stats);
    }
    return summary;
}

void print_monte_carlo_summary(
    const MonteCarloSummary& summary,
    const std::vector<std::string>& datetimes
) {
    for (size_t i = 0; i < summary.fields.size(); ++i) {
        print_monte_carlo_stats(summary.stats[i], datetimes, summary.fields[i]);
        std::cout << std::endl;
    }
}
