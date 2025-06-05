#include <iostream>
#include "common.h"
#include "lp_strategy.h"
#include "pool_data_fetching.h"
#include "utils/conversions.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <apiSubgraphs> <idSubgraphs> <poolAddress>" << std::endl;
        return 1; 
    }
    std::string apiSubgraphs = argv[1];
    std::string idSubgraphs = argv[2];
    std::string poolAddress = argv[3];
    std::string startDate = "2025-05-17";
    std::string endDate = "2025-05-19";
	
    auto pool_data_opt = fetch_pool_data(apiSubgraphs, idSubgraphs, poolAddress, startDate, endDate);
    if (!pool_data_opt.has_value()) {
        std::cerr << "Failed to fetch pool data." << std::endl;
        return 1;
    }
    const PoolData& pool_data = pool_data_opt.value();

    // Determine start price and set strategy range
    double start_price = pool_data.price.empty() ? 0.0 : pool_data.price.front();
    LPStrategy strategy;
    strategy.lower_price = start_price * 0.98;
    strategy.upper_price = start_price * 1.02;
    strategy.amount_token0 = 1;
    double Lx = strategy.amount_token0 * (std::sqrt(start_price) * std::sqrt(strategy.upper_price)) / (std::sqrt(strategy.upper_price) - std::sqrt(start_price));
    strategy.amount_token1 = Lx * (std::sqrt(start_price)-std::sqrt(strategy.lower_price));
    strategy.reinvest_fees = false; // Set to true if you want to reinvest fees

    // Simulate strategy
    LPStrategyResults results = simulateLPStrategy(pool_data, strategy);
    // Output results
    printResults(pool_data, results);

    // Convert fees0 (token0) and fees1 (token1) to USD using the conversion utility
    std::vector<double> fees0_usd = convert_to_usd(
        results.fees0_acc,
        pool_data.token0_symbol,
        results.datetime,
        apiSubgraphs,
        idSubgraphs
    );
    std::vector<double> fees1_usd = convert_to_usd(
        results.fees1_acc,
        pool_data.token1_symbol,
        results.datetime,
        apiSubgraphs,
        idSubgraphs
    );

    // Compute IL in USD using the utility function
    std::vector<double> il_usd = compute_impermanent_loss_usd(
        results,
        results.datetime,
        apiSubgraphs,
        idSubgraphs
    );

    // Output side-by-side: Datetime | Price | Price Range | Fees (USD) | IL (USD) | Net Profit (USD)
    std::cout << "\nDatetime\tPrice (token1)\tLower\tUpper\tFees Acc. (USD)\tIL (USD)\tNet Profit (USD)\n";
    for (size_t i = 0; i < fees0_usd.size(); ++i) {
        double fees_usd = fees0_usd[i] + fees1_usd[i];
        double net = fees_usd + il_usd[i];
        std::cout << results.datetime[i] << "\t"
                  << pool_data.price[i] << "\t"
                  << strategy.lower_price << "\t"
                  << strategy.upper_price << "\t"
                  << results.fees0_acc[i] + results.fees1_acc[i] << "\t"
                  << fees_usd << "\t"
                  << il_usd[i] << "\t"
                  << net << "\n";
    }

    return 0;
}
