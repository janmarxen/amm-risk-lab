#include <iostream>
#include "common.h"
#include "lp_simulation.h"
#include "pool_data_fetching.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <apiSubgraphs> <idSubgraphs> <poolAddress>" << std::endl;
        return 1; 
    }
    std::string apiSubgraphs = argv[1];
    std::string idSubgraphs = argv[2];
    std::string poolAddress = argv[3];
    std::string startDate = "2025-05-17";
    std::string endDate = "2025-05-17";
	
    auto pool_data_opt = fetch_pool_series(apiSubgraphs, idSubgraphs, poolAddress, startDate, endDate);
    if (!pool_data_opt.has_value()) {
        std::cerr << "Failed to fetch pool data." << std::endl;
        return 1;
    }
    const PoolSeries& pool_data = pool_data_opt.value();

    // Print the fee rate of the pool
    std::cout << "Fee rate of the pool: " << pool_data.fee_tier << std::endl;

    // Determine start price and set strategy range
    double start_price = pool_data.price.size() > 0 ? pool_data.price(0, 0) : 0.0;
    double lower_price = start_price * 0.98;
    double upper_price = start_price * 1.02;
    double amount_token0 = 1;
    double Lx = amount_token0 * (std::sqrt(start_price) * std::sqrt(upper_price)) / (std::sqrt(upper_price) - std::sqrt(start_price));
    double amount_token1 = Lx * (std::sqrt(start_price) - std::sqrt(lower_price));
    bool reinvest_fees = false;

    UniswapV3LPStrategy strategy(lower_price, upper_price, amount_token0, amount_token1, reinvest_fees);

    UniswapV3LPSimulator simulator(apiSubgraphs, idSubgraphs);
    LPSimulationResults results = simulator.simulate(pool_data, strategy);

    // simulator.printResults(pool_data, results);

    // Output side-by-side: Datetime | Price | Price Range | Fees (USD) | IL (USD) | Net Profit (USD)
    std::cout << "\nDatetime\tPrice (token1)\tLower\tUpper\tFees Acc. (USD)\tIL (USD)\tNet Profit (USD)\n";
    int n_points = pool_data.n_points;
    for (int i = 0; i < n_points; ++i) {
        double fees_usd = results.feesUSD_acc(0, i); // Only first path for display
        double il_usd = results.il_usd(0, i);
        double net = fees_usd + il_usd;
        std::cout << results.datetimes[i] << "\t"
                  << pool_data.price(0, i) << "\t"
                  << strategy.lower_price << "\t"
                  << strategy.upper_price << "\t"
                  << fees_usd << "\t"
                  << il_usd << "\t"
                  << net << "\n";
    }

    return 0;
}
