#include "lp_strategy.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include "utils/conversions.h"

LPStrategyResults simulateLPStrategy(
    const PoolData& pool,
    const LPStrategy& strategy
) {
    if (pool.data_points == 0) return {};

    size_t n = pool.data_points;
    LPStrategyResults results;
    results.token0_symbol = pool.token0_symbol;
    results.token1_symbol = pool.token1_symbol;
    results.start_amount_token0 = strategy.amount_token0;
    results.start_amount_token1 = strategy.amount_token1;
    results.datetime = pool.datetime;
    results.user_liquidity.resize(n);
    results.token0_amounts.resize(n);
    results.token1_amounts.resize(n);
    results.fees0_acc.resize(n);
    results.fees1_acc.resize(n);

    double sqrt_a = std::sqrt(strategy.lower_price);
    double sqrt_b = std::sqrt(strategy.upper_price);
    double sqrt_b_minus_a = sqrt_b - sqrt_a;

    double L = 0;
    double x = strategy.amount_token0;
    double y = strategy.amount_token1;
    double cum_fee0 = 0.0;
    double cum_fee1 = 0.0;
    double feeGrowth0 = 0.0;
    double feeGrowth1 = 0.0;

    // Initialize L in the first iteration
    double L_x = x * (sqrt(pool.price[0]) * sqrt_b) / (sqrt_b - sqrt(pool.price[0]));
    double L_y = y / (sqrt(pool.price[0]) - sqrt_a);
    L = std::min(L_x, L_y);

    for (size_t i = 0; i < n; ++i) {
        double P = pool.price[i];
        double sqrt_P = std::sqrt(P);

        // Calculate token amounts based on current price
        if (P <= strategy.lower_price) {
            x = L * (sqrt_b_minus_a) / (sqrt_a * sqrt_b);
            y = 0.0;
        } else if (P >= strategy.upper_price) {
            x = 0.0;
            y = L * (sqrt_b_minus_a);
        } else {
            x = L * (sqrt_b - sqrt_P) / (sqrt_P * sqrt_b);
            y = L * (sqrt_P - sqrt_a);
        }

        // Update fee growth
        feeGrowth0 += (pool.volume0[i] * pool.fee_tier) / pool.liquidity[i];
        feeGrowth1 += (pool.volume1[i] * pool.fee_tier) / pool.liquidity[i];

        // Calculate cumulative fees based on liquidity and fee growth
        double fees0 = L * feeGrowth0;
        double fees1 = L * feeGrowth1;

        if(strategy.reinvest_fees) {
            // Reinvest fees
            x += fees0;
            y += fees1;
            // Recalculate L for new amounts
            if (P <= strategy.lower_price) {
                L = x * (sqrt_a * sqrt_b) / (sqrt_b_minus_a);
            } else if (P >= strategy.upper_price) {
                L = y / (sqrt_b_minus_a);
            } else {
                double L_x_new = x * (sqrt_P * sqrt_b) / (sqrt_b - sqrt_P);
                double L_y_new = y / (sqrt_P - sqrt_a);
                L = std::min(L_x_new, L_y_new);
            }
        } else {
            // Withdraw fees
            cum_fee0 += fees0;
            cum_fee1 += fees1;
        }

        // Store results
        results.token0_amounts[i] = x;
        results.token1_amounts[i] = y;
        results.fees0_acc[i] = cum_fee0;
        results.fees1_acc[i] = cum_fee1;

        results.user_liquidity[i] = L;
    }
    results.final_amount_token0 = results.token0_amounts.back();
    results.final_amount_token1 = results.token1_amounts.back();
    return results;
}

std::vector<double> compute_impermanent_loss_usd(
    const LPStrategyResults& results,
    const std::vector<std::string>& datetime,
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs
) {
    // Compute LP value in USD at each time step
    std::vector<double> lp_value_token0 = results.token0_amounts;
    std::vector<double> lp_value_token1 = results.token1_amounts;

    std::vector<double> lp_value_usd_token0 = convert_to_usd(
        lp_value_token0, results.token0_symbol, datetime, apiSubgraphs, idSubgraphs
    );
    std::vector<double> lp_value_usd_token1 = convert_to_usd(
        lp_value_token1, results.token1_symbol, datetime, apiSubgraphs, idSubgraphs
    );

    std::vector<double> lp_value_usd(lp_value_usd_token0.size());
    for (size_t i = 0; i < lp_value_usd.size(); ++i) {
        lp_value_usd[i] = lp_value_usd_token0[i] + lp_value_usd_token1[i];
    }

    // Compute HODL value in USD at each time step (using initial amounts)
    std::vector<double> hodl_token0(lp_value_usd.size(), results.start_amount_token0);
    std::vector<double> hodl_token1(lp_value_usd.size(), results.start_amount_token1);

    std::vector<double> hodl_usd_token0 = convert_to_usd(
        hodl_token0, results.token0_symbol, datetime, apiSubgraphs, idSubgraphs
    );
    std::vector<double> hodl_usd_token1 = convert_to_usd(
        hodl_token1, results.token1_symbol, datetime, apiSubgraphs, idSubgraphs
    );

    std::vector<double> hodl_value_usd(lp_value_usd.size());
    for (size_t i = 0; i < hodl_value_usd.size(); ++i) {
        hodl_value_usd[i] = hodl_usd_token0[i] + hodl_usd_token1[i];
    }

    // IL in USD = LP value in USD - HODL value in USD
    std::vector<double> il_usd(lp_value_usd.size());
    for (size_t i = 0; i < il_usd.size(); ++i) {
        il_usd[i] = lp_value_usd[i] - hodl_value_usd[i];
    }
    return il_usd;
}

// Outputs the simulation results with scientific notation
void printResults(const PoolData& pool_data, const LPStrategyResults& results) {
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Datetime\tPrice\tUserLiquidity\tToken0\tToken1\tFees0\tFees1\n";
    for (size_t i = 0; i < pool_data.data_points; ++i) {
        std::cout << results.datetime[i] << "\t"
                  << pool_data.price[i] << "\t"
                  << results.user_liquidity[i] << "\t"
                  << results.token0_amounts[i] << "\t"
                  << results.token1_amounts[i] << "\t"
                  << results.fees0_acc[i] << "\t"
                  << results.fees1_acc[i] << "\n";
    }
}