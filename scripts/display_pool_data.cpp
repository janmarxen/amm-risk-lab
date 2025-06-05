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
    std::string startDate = "2025-03-27";
    std::string endDate = "2025-03-27";
	
	auto pool_data = fetch_pool_data(apiSubgraphs, idSubgraphs, poolAddress, startDate, endDate);

    if (pool_data) {
        // Access the structured data
        std::cout << "Pool: " << pool_data->token0_symbol << "/" << pool_data->token1_symbol 
                  << " (Fee: " << pool_data->fee_tier << "%) " 
				  << " (Data Points: " << pool_data->data_points << ")" << std::endl;
        
        for (size_t i = 0; i < pool_data->datetime.size(); ++i) {
            std::cout << pool_data->datetime[i] 
                      << " | Price: " << pool_data->price[i]
                      << " | Volume (USD): $" << pool_data->volumeUSD[i]
                      << " | Liquidity: $" << pool_data->liquidity[i]
					  << " | Fees (USD): $" << pool_data->volumeUSD[i]*pool_data->fee_tier/1000
                      << std::endl;
        }
    } else {
        std::cout << "Failed to fetch pool data." << std::endl;
    }

    return 0;
}


