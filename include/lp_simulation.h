#pragma once

#include "common.h"
#include <vector>
#include <string>
#include <Eigen/Dense>

/**
 * @brief Abstract base class for liquidity provider strategies.
 */
class LPStrategy {
public:
    virtual ~LPStrategy() = default;
    virtual std::string protocol() const = 0;
};

/**
 * @brief Strategy for Uniswap V3 liquidity provision.
 */
class UniswapV3LPStrategy : public LPStrategy {
public:
    double lower_price;
    double upper_price;
    double amount_token0;
    double amount_token1;
    bool reinvest_fees;

    UniswapV3LPStrategy(double lower, double upper, double amt0, double amt1, bool reinvest)
        : lower_price(lower), upper_price(upper), amount_token0(amt0), amount_token1(amt1), reinvest_fees(reinvest) {}

    std::string protocol() const override { return "UniswapV3"; }
};

/**
 * @brief Holds the results of simulating an LP strategy (supports multiple paths, Eigen-based, USD only).
 */
struct LPSimulationResults {
    std::string token0_symbol;
    std::string token1_symbol;
    double start_amount_token0;
    double start_amount_token1;
    Eigen::VectorXd final_amount_token0; // n_paths
    Eigen::VectorXd final_amount_token1; // n_paths
    std::vector<std::string> datetimes;  // length n_points (shared for all paths)
    Eigen::MatrixXd user_liquidity;      // n_paths x n_points
    Eigen::MatrixXd token0_amounts;      // n_paths x n_points
    Eigen::MatrixXd token1_amounts;      // n_paths x n_points
    Eigen::MatrixXd feesUSD_acc;         // n_paths x n_points
    Eigen::MatrixXd il_usd;              // n_paths x n_points, impermanent loss in USD
};

/**
 * @brief Abstract base class for LP simulators.
 */
class LPSimulator {
public:
    LPSimulator() = default;
    virtual ~LPSimulator() = default;

    virtual LPSimulationResults simulate(
        const PoolSeries& pool,
        const LPStrategy& strategy
    ) const = 0;

    virtual void printResults(const PoolSeries& pool_data, const LPSimulationResults& results) const;
};

/**
 * @brief Simulator for Uniswap V3 LP strategies.
 */
class UniswapV3LPSimulator : public LPSimulator {
public:
    std::string apiSubgraphs;
    std::string idSubgraphs;

    UniswapV3LPSimulator(const std::string& apiSubgraphs_, const std::string& idSubgraphs_)
        : apiSubgraphs(apiSubgraphs_), idSubgraphs(idSubgraphs_) {}

    LPSimulationResults simulate(
        const PoolSeries& pool,
        const LPStrategy& strategy
    ) const override;

    // Overload: Simulate using owner address (fetches strategy from subgraph)
    LPSimulationResults simulate(
        const std::string& poolAddress,
        const std::string& ownerAddress,
        const std::string& startDate,
        const std::string& endDate
    ) const;

    void printResults(const PoolSeries& pool_data, const LPSimulationResults& results) const override;

    void compute_impermanent_loss_usd(
        LPSimulationResults& results,
        const PoolSeries& pool
    ) const;
};

/**
 * @brief Simulate an LP strategy on a pool.
 */
LPSimulationResults simulateLPStrategy(
    const PoolSeries& pool,
    const LPStrategy& strategy,
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs
);

/**
 * @brief Compute impermanent loss in USD over time.
 * @return Matrix of impermanent loss values in USD (n_paths x n_points).
 */
Eigen::MatrixXd compute_impermanent_loss_usd(
    const LPSimulationResults& results
);

/**
 * @brief Print the simulation results.
 */
void printResults(const PoolSeries& pool_data, const LPSimulationResults& results);
