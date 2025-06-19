#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "pool_data_fetching.h"
#include "utils/conversions.h"
#include "utils/subgraph_utils.h"

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

    // We'll use token0 as the currency to convert to USD
    std::vector<double> values; 
    values.reserve(pool_data.price.size());
    for (size_t i = 0; i < pool_data.price.size(); ++i) {
        values.push_back(1.0); // 1 unit of token0 per row
    }
    const std::vector<std::string>& datetimes = pool_data.datetimes;
    std::string currency_symbol = pool_data.token1_symbol;

    // Convert to USD
    std::vector<double> usd_values = convert_to_usd(
        values,
        currency_symbol,
        datetimes,
        apiSubgraphs,
        idSubgraphs
    );

    // Print table header
    std::cout << std::setw(20) << "Datetime"
              << std::setw(15) << "Value"
              << std::setw(15) << "Currency"
              << std::setw(15) << "USD Value"
              << std::endl;

    // Print conversion table
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << std::setw(20) << datetimes[i]
                  << std::setw(15) << values[i]
                  << std::setw(15) << currency_symbol
                  << std::setw(15) << usd_values[i]
                  << std::endl;
    }

    return 0;
}
