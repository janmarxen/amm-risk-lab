#pragma once

#include <vector>
#include <string>
#include "common.h"
#include "utils/conversions.h"

/// Represents a liquidity provider (LP) strategy for a pool.
struct LPStrategy {
    double lower_price;
    double upper_price;
    double amount_token0;
    double amount_token1;
    bool reinvest_fees = false;
};

/// Results of simulating an LP strategy.
struct LPStrategyResults {
    std::vector<std::string> datetime;
    std::vector<double> user_liquidity;
    std::string token0_symbol;
    std::string token1_symbol;
    double start_amount_token0;
    double start_amount_token1;
    double final_amount_token0;
    double final_amount_token1;
    std::vector<double> token0_amounts;
    std::vector<double> token1_amounts;
    std::vector<double> fees0_acc;
    std::vector<double> fees1_acc;
};

LPStrategyResults simulateLPStrategy(
    const PoolData& pool,
    const LPStrategy& strategy
);

void printResults(const PoolData& pool_data, 
    const LPStrategyResults& results);

/// Computes the impermanent loss in USD for each time step using conversion utilities.
/// Returns a vector of IL in USD, same length as results.datetime.
std::vector<double> compute_impermanent_loss_usd(
    const LPStrategyResults& results,
    const std::vector<std::string>& datetime,
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs
);
