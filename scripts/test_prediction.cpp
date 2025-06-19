#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "common.h"
#include "price_generation.h"
#include "PLV_model/prediction.h"
#include "lp_simulation.h"
#include "monte_carlo.h"
#include "pool_data_fetching.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <apiSubgraphs> <idSubgraphs> <poolAddress>" << std::endl;
        return 1; 
    }
    std::string apiSubgraphs = argv[1];
    std::string idSubgraphs = argv[2];
    std::string poolAddress = argv[3];
    // GBM Parameters
    double S0 = 100.0;
    double mu = 0.0;
    double sigma = 0.001;
    std::string start_datetime = "2025-01-01T00:00:00";
    std::string end_datetime = "2025-01-01T12:00:00";
    int n_paths = 100000; // Number of paths to simulate
    // Pool parameters

    // Fetch token names from pool address
    auto token_names_opt = get_pool_token_names(apiSubgraphs, idSubgraphs, poolAddress);
    if (!token_names_opt) {
        std::cerr << "Failed to fetch token names for pool: " << poolAddress << std::endl;
        return 1;
    }
    std::string token0 = token_names_opt->first;
    std::string token1 = token_names_opt->second;
    double fee_tier = 0.003;

    // Generate price paths
    PriceSeries price_series = simulateGBMSeries(
        S0, mu, sigma, start_datetime, end_datetime, n_paths, false, 0, 0, false
    );

    // Predict volume and liquidity (dummy)
    PoolSeries pool_usd = dummy_predict_plv(price_series, token0, token1, fee_tier);

    // Set up a Uniswap V3 LP strategy for each path
    double lower_price = S0 * 0.98;
    double upper_price = S0 * 1.02;
    double amount_token0 = 1.0;
    double Lx = amount_token0 * (std::sqrt(S0) * std::sqrt(upper_price)) / (std::sqrt(upper_price) - std::sqrt(S0));
    double amount_token1 = Lx * (std::sqrt(S0) - std::sqrt(lower_price));
    bool reinvest_fees = false;
    UniswapV3LPStrategy strategy(lower_price, upper_price, amount_token0, amount_token1, reinvest_fees);

    // Simulate that strategy
    UniswapV3LPSimulator simulator(apiSubgraphs, idSubgraphs);

    std::cout << "Simulating Uniswap V3 LP strategy for pool... " << std::endl;

    // Run simulation
    LPSimulationResults sim_results = simulator.simulate(pool_usd, strategy);

    // Print simulation results for all paths
    // simulator.printResults(pool_usd, sim_results);

    // Monte Carlo statistics for all fields (including il_usd)
    MonteCarloSummary summary = summarize_all_stats(sim_results, sim_results.datetimes);
    std::cout << "Monte Carlo summary for all fields:" << std::endl;
    print_monte_carlo_summary(summary, sim_results.datetimes);

    return 0;
}
