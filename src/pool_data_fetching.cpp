#include "pool_data_fetching.h"
#include "common.h"
#include "utils/time_utils.h"
#include "utils/subgraph_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include <curl/curl.h>

using json = nlohmann::json;

// Function to parse the JSON response into PoolData
std::optional<PoolData> parse_pool_data(const json& json_data) {
    try {
        PoolData data;

        // Parse pool information
        if (json_data.contains("data") && json_data["data"].contains("pool")) {
            const auto& pool = json_data["data"]["pool"];
            data.token0_symbol = pool["token0"]["symbol"].get<std::string>();
            data.token1_symbol = pool["token1"]["symbol"].get<std::string>();
            data.fee_tier = std::stod(pool["feeTier"].get<std::string>()) / 10000; // Convert from basis points
        }

        // Parse hourly data
        if (json_data.contains("data") && json_data["data"].contains("poolHourDatas")) {
            for (const auto& hour_data : json_data["data"]["poolHourDatas"]) {
                time_t timestamp = hour_data["periodStartUnix"].get<time_t>();
                data.datetime.push_back(unix_to_datetime(timestamp));
                data.price.push_back(std::stod(hour_data["token0Price"].get<std::string>()));
                data.volume0.push_back(std::stod(hour_data["volumeToken0"].get<std::string>()));
                data.volume1.push_back(std::stod(hour_data["volumeToken1"].get<std::string>()));
                data.volumeUSD.push_back(std::stod(hour_data["volumeUSD"].get<std::string>()));
                data.liquidity.push_back(std::stod(hour_data["liquidity"].get<std::string>()));
            }
            // Set the data_points count
            data.data_points = data.datetime.size();
        }

        return data;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing pool data: " << e.what() << std::endl;
        return std::nullopt;
    }
}


std::optional<PoolData> fetch_pool_data(const std::string& apiSubgraphs, 
    const std::string& idSubgraphs, 
    const std::string& poolAddress, 
    const std::string& startDate, 
    const std::string& endDate) {

    time_t start_ts = date_to_unix(startDate);
    time_t end_ts = date_to_unix(endDate) + 86399; // end of day (23:59:59)

    // Construct the query string
    std::string graphqlQuery = R"({
    pool(id: ")" + poolAddress + R"(") {
        id
        token0 {
        symbol
        }
        token1 {
        symbol
        }
        feeTier
    }
    poolHourDatas(
        orderBy: periodStartUnix,
        orderDirection: asc,
        where: { pool: ")" + poolAddress + R"(", periodStartUnix_gte: )" + std::to_string(start_ts) + R"(, periodStartUnix_lte: )" + std::to_string(end_ts) + R"( }
    ) {
        periodStartUnix
        liquidity
        volumeToken0
        volumeToken1
        volumeUSD
        token0Price
        feesUSD
    }
    })";

    // Use modularized GraphQL POST helper
    auto json_data_opt = graphql_post(apiSubgraphs, idSubgraphs, graphqlQuery);
    if (!json_data_opt.has_value()) {
        return std::nullopt;
    }
    return parse_pool_data(json_data_opt.value());
}
