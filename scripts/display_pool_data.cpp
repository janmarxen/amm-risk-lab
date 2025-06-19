#include <iostream>
#include "pool_data_fetching.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <apiSubgraphs> <idSubgraphs> <poolAddress>" << std::endl;
        return 1; 
    }
    std::string apiSubgraphs = argv[1];
    std::string idSubgraphs = argv[2];
    std::string poolAddress = argv[3];
    std::string startDate = "2025-05-01";
    std::string endDate = "2025-06-01";
	
    auto pool_series = fetch_pool_series(apiSubgraphs, idSubgraphs, poolAddress, startDate, endDate);

    if (pool_series) {
        // Access the structured data using the PoolSeries class
        std::cout << "Pool: " << pool_series->token0_symbol << "/" << pool_series->token1_symbol 
                  << " (Fee: " << pool_series->fee_tier << "%) " 
                  << " (Data Points: " << pool_series->n_points << ")" << std::endl;

        // for (int i = 0; i < pool_series->n_points; ++i) {
        //     std::cout << pool_series->datetimes[i] 
        //               << " | Price: " << pool_series->price[i]
        //               << " | Volume (USD): $" << pool_series->volumeUSD[i]
        //               << " | Liquidity: $" << pool_series->liquidity[i]
        //               << " | Fees (USD): $" << pool_series->volumeUSD[i] * pool_series->fee_tier / 1000
        //               << std::endl;
        // }

        // Compose output filename based on token symbols
        std::string out_filename = "pool_data_" + pool_series->token0_symbol + "_" + pool_series->token1_symbol + ".txt";
        write_poolseries_to_txt(out_filename, *pool_series);
        std::cout << "Pool data written to " << out_filename << std::endl;
    } else {
        std::cout << "Failed to fetch pool data." << std::endl;
    }

    return 0;
}


