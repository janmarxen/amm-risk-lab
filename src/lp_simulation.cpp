#include "lp_simulation.h"
#include "pool_data_fetching.h"
#include "common.h"
#include "utils/conversions.h"
#include "utils/time_utils.h"
#include "utils/subgraph_utils.h"
#include <cmath>
#include <iostream>
#include <iomanip>

void LPSimulator::printResults(const PoolSeries& pool_data, const LPSimulationResults& results) const {
    int n_paths = results.user_liquidity.rows();
    int n_points = results.user_liquidity.cols();
    for (int path = 0; path < n_paths; ++path) {
        std::cout << "=== Path " << path << " (USD) ===\n";
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "Datetime\tPrice\tUserLiquidity\tToken0\tToken1\tFeesUSD\tIL_USD\n";
        for (int i = 0; i < n_points; ++i) {
            std::cout << results.datetimes[i] << "\t"
                      << pool_data.price(path, i) << "\t"
                      << results.user_liquidity(path, i) << "\t"
                      << results.token0_amounts(path, i) << "\t"
                      << results.token1_amounts(path, i) << "\t"
                      << results.feesUSD_acc(path, i) << "\t"
                      << results.il_usd(path, i) << "\n";
        }
    }
}

void UniswapV3LPSimulator::printResults(const PoolSeries& pool, const LPSimulationResults& results) const {
    LPSimulator::printResults(pool, results);
}

LPSimulationResults UniswapV3LPSimulator::simulate(
    const PoolSeries& pool,
    const LPStrategy& strategy
) const {
    if (pool.n_points == 0) return {};

    const UniswapV3LPStrategy* uni3_strategy = dynamic_cast<const UniswapV3LPStrategy*>(&strategy);
    if (!uni3_strategy) {
        throw std::invalid_argument("Strategy must be UniswapV3LPStrategy for UniswapV3LPSimulator");
    }

    int n_paths = pool.n_paths > 0 ? pool.n_paths : 1;
    int n_points = pool.n_points;

    LPSimulationResults results;
    results.token0_symbol = pool.token0_symbol;
    results.token1_symbol = pool.token1_symbol;
    results.start_amount_token0 = uni3_strategy->amount_token0;
    results.start_amount_token1 = uni3_strategy->amount_token1;
    results.datetimes = pool.datetimes;
    results.user_liquidity = Eigen::MatrixXd::Zero(n_paths, n_points);
    results.token0_amounts = Eigen::MatrixXd::Zero(n_paths, n_points);
    results.token1_amounts = Eigen::MatrixXd::Zero(n_paths, n_points);
    results.feesUSD_acc = Eigen::MatrixXd::Zero(n_paths, n_points);
    results.final_amount_token0 = Eigen::VectorXd::Zero(n_paths);
    results.final_amount_token1 = Eigen::VectorXd::Zero(n_paths);

    for (int path = 0; path < n_paths; ++path) {
        double sqrt_a = std::sqrt(uni3_strategy->lower_price);
        double sqrt_b = std::sqrt(uni3_strategy->upper_price);
        double sqrt_b_minus_a = sqrt_b - sqrt_a;

        double L = 0;
        double x = uni3_strategy->amount_token0;
        double y = uni3_strategy->amount_token1;
        double cum_fees_usd = 0.0;

        double price0 = pool.price(path, 0);
        double L_x = x * (std::sqrt(price0) * sqrt_b) / (sqrt_b - std::sqrt(price0));
        double L_y = y / (std::sqrt(price0) - sqrt_a);
        L = std::min(L_x, L_y);

        for (int i = 0; i < n_points; ++i) {
            double P = pool.price(path, i);
            double sqrt_P = std::sqrt(P);
            bool in_range = (P > uni3_strategy->lower_price && P < uni3_strategy->upper_price);

            // Token balances
            if (P <= uni3_strategy->lower_price) {
                x = L * (sqrt_b_minus_a) / (sqrt_a * sqrt_b);
                y = 0.0;
            } else if (P >= uni3_strategy->upper_price) {
                x = 0.0;
                y = L * (sqrt_b_minus_a);
            } else {
                x = L * (sqrt_b - sqrt_P) / (sqrt_P * sqrt_b);
                y = L * (sqrt_P - sqrt_a);
            }

            if (in_range) {
                double share = L / pool.liquidity(path, i);
                double fees_usd = pool.volumeUSD(path, i) * pool.fee_tier * share;
                cum_fees_usd += fees_usd;
            }

            results.token0_amounts(path, i) = x;
            results.token1_amounts(path, i) = y;
            results.user_liquidity(path, i) = L;
            results.feesUSD_acc(path, i) = cum_fees_usd;
        }
        results.final_amount_token0(path) = results.token0_amounts.row(path).tail(1)(0);
        results.final_amount_token1(path) = results.token1_amounts.row(path).tail(1)(0);
    }

    // Compute impermanent loss in USD for all paths and store in results
    compute_impermanent_loss_usd(results, pool);
    return results;
}

LPSimulationResults UniswapV3LPSimulator::simulate(
    const std::string& poolAddress,
    const std::string& ownerAddress,
    const std::string& startDate,
    const std::string& endDate
) const {
    // Fetch the owner's first position snapshot for the pool
    time_t start_ts = date_to_unix(startDate);
    time_t end_ts = date_to_unix(endDate) + 86399;
    std::string graphqlQuery = R"({
      positionSnapshots(
        where: {
          owner: \"" + ownerAddress + R"\",
          pool: \"" + poolAddress + R"\",
          timestamp_gte: " + std::to_string(start_ts) + R"(,
          timestamp_lte: " + std::to_string(end_ts) + R"()
        },
        orderBy: timestamp,
        orderDirection: asc,
        first: 1
      ) {
        liquidity
        depositedToken0
        depositedToken1
        withdrawnToken0
        withdrawnToken1
        lowerTick
        upperTick
      }
      pool(id: \"" + poolAddress + R"\") {
        token0 { symbol }
        token1 { symbol }
        feeTier
      }
    })";
    auto json_data_opt = run_subgraph_query(apiSubgraphs, idSubgraphs, graphqlQuery);
    if (!json_data_opt.has_value()) {
        throw std::runtime_error("Failed to fetch owner's first position snapshot.");
    }
    const auto& json_data = json_data_opt.value();
    if (!(json_data.contains("data") && json_data["data"].contains("positionSnapshots") && json_data["data"]["positionSnapshots"].size() > 0)) {
        throw std::runtime_error("No position snapshot found for owner.");
    }
    const auto& snap = json_data["data"]["positionSnapshots"][0];
    double amount_token0 = std::stod(snap["depositedToken0"].get<std::string>()) - std::stod(snap["withdrawnToken0"].get<std::string>());
    double amount_token1 = std::stod(snap["depositedToken1"].get<std::string>()) - std::stod(snap["withdrawnToken1"].get<std::string>());
    int lowerTick = snap["lowerTick"].is_null() ? 0 : std::stoi(snap["lowerTick"].get<std::string>());
    int upperTick = snap["upperTick"].is_null() ? 0 : std::stoi(snap["upperTick"].get<std::string>());
    double lower_price = std::pow(1.0001, lowerTick);
    double upper_price = std::pow(1.0001, upperTick);
    bool reinvest_fees = false;
    // Now fetch pool data and simulate
    auto pool_data_opt = fetch_pool_series(apiSubgraphs, idSubgraphs, poolAddress, startDate, endDate);
    if (!pool_data_opt.has_value()) {
        throw std::runtime_error("Failed to fetch pool data for simulation.");
    }
    const PoolSeries& pool_data = pool_data_opt.value();
    UniswapV3LPStrategy strategy(lower_price, upper_price, amount_token0, amount_token1, reinvest_fees);
    return simulate(pool_data, strategy);
}

void UniswapV3LPSimulator::compute_impermanent_loss_usd(
    LPSimulationResults& results,
    const PoolSeries& pool
) const {
    int n_paths = results.token0_amounts.rows();
    int n_points = results.token0_amounts.cols();
    results.il_usd = Eigen::MatrixXd::Zero(n_paths, n_points);

    const std::vector<std::string>& datetimes = results.datetimes;

    // Fetch price series for token0->USDC and token1->USDC once
    std::vector<double> token0_to_usdc(n_points, 1.0);
    std::vector<double> token1_to_usdc(n_points, 1.0);

    if (results.token0_symbol != "USDC") {
        auto price_info_opt = fetch_price_datetime_series(apiSubgraphs, idSubgraphs, 
            get_pool_address(results.token0_symbol + "_USDC", idSubgraphs),
            datetimes.front().substr(0, 10), datetimes.back().substr(0, 10));
        if (price_info_opt.has_value()) {
            const auto& price_info = price_info_opt.value();
            for (int i = 0; i < n_points; ++i) {
                token0_to_usdc[i] = price_info.prices(0, i);
            }
        }
    }

    if (results.token1_symbol != "USDC") {
        auto price_info_opt = fetch_price_datetime_series(apiSubgraphs, idSubgraphs, 
            get_pool_address(results.token1_symbol + "_USDC", idSubgraphs),
            datetimes.front().substr(0, 10), datetimes.back().substr(0, 10));
        if (price_info_opt.has_value()) {
            const auto& price_info = price_info_opt.value();
            for (int i = 0; i < n_points; ++i) {
                token1_to_usdc[i] = price_info.prices(0, i);
            }
        }
    }

    // Map token prices as Eigen row vectors
    Eigen::RowVectorXd token0_to_usdc_row = Eigen::Map<const Eigen::VectorXd>(token0_to_usdc.data(), n_points).transpose();
    Eigen::RowVectorXd token1_to_usdc_row = Eigen::Map<const Eigen::VectorXd>(token1_to_usdc.data(), n_points).transpose();

    // Compute LP value in USD for all paths
    Eigen::MatrixXd lp_value_usd = results.token0_amounts.array().rowwise() * token0_to_usdc_row.array()
                                + results.token1_amounts.array().rowwise() * token1_to_usdc_row.array();

    // Compute HODL value in USD for all paths
    Eigen::RowVectorXd hodl_value_usd = results.start_amount_token0 * token0_to_usdc_row
                                    + results.start_amount_token1 * token1_to_usdc_row;

    // Subtract HODL from LP to get IL
    results.il_usd = lp_value_usd.rowwise() - hodl_value_usd;
}


void printResults(const PoolSeries& pool_data, const LPSimulationResults& results, 
                  const std::string& apiSubgraphs, const std::string& idSubgraphs) {
    UniswapV3LPSimulator sim(apiSubgraphs, idSubgraphs);
    sim.printResults(pool_data, results);
}
