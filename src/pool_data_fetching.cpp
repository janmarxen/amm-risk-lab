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
#include <fstream>
#include <set>

using json = nlohmann::json;


std::optional<PoolSeries> parse_pool_series(const json& json_data, bool parse_pool_info = true) {
    try {
        PoolSeries data;
        data.n_paths = 1; // Only one path when fetching from API

        if (parse_pool_info && json_data.contains("data") && json_data["data"].contains("pool")) {
            const auto& pool = json_data["data"]["pool"];
            data.token0_symbol = pool["token0"]["symbol"].get<std::string>();
            data.token1_symbol = pool["token1"]["symbol"].get<std::string>();
            data.fee_tier = std::stod(pool["feeTier"].get<std::string>()) / 10000;
        }

        if (json_data.contains("data") && json_data["data"].contains("poolHourDatas")) {
            const auto& hour_datas = json_data["data"]["poolHourDatas"];
            int n = static_cast<int>(hour_datas.size());
            data.n_points = n;
            data.price = Eigen::MatrixXd::Zero(1, n);
            data.volumeUSD = Eigen::MatrixXd::Zero(1, n);
            data.liquidity = Eigen::MatrixXd::Zero(1, n);

            for (int i = 0; i < n; ++i) {
                const auto& hour_data = hour_datas[i];
                time_t timestamp = hour_data["periodStartUnix"].get<time_t>();
                data.datetimes.push_back(unix_to_datetime(timestamp));
                data.price(0, i) = std::stod(hour_data["token0Price"].get<std::string>());
                data.volumeUSD(0, i) = std::stod(hour_data["volumeUSD"].get<std::string>());
                data.liquidity(0, i) = std::stod(hour_data["liquidity"].get<std::string>());
            }
        }

        return data;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing pool data: " << e.what() << std::endl;
        return std::nullopt;
    }
}


std::optional<PoolSeries> fetch_pool_series(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress,
    const std::string& startDate,
    const std::string& endDate
) {
    time_t start_ts = date_to_unix(startDate);
    time_t end_ts = date_to_unix(endDate) + 86399;

    PoolSeries full_data;
    int skip = 0;
    const int batch_size = 1000;

    while (true) {
        // Construct query with skip
        std::string graphqlQuery = R"({
        pool(id: ")" + poolAddress + R"(") {
            id
            token0 { symbol }
            token1 { symbol }
            feeTier
        }
        poolHourDatas(
            first: 1000,
            skip: )" + std::to_string(skip) + R"(,
            orderBy: periodStartUnix,
            orderDirection: asc,
            where: {
                pool: ")" + poolAddress + R"(",
                periodStartUnix_gte: )" + std::to_string(start_ts) + R"(,
                periodStartUnix_lte: )" + std::to_string(end_ts) + R"(
            }
        ) {
            periodStartUnix
            liquidity
            volumeUSD
            token0Price
            feesUSD
        }
        })";

        auto json_data_opt = run_subgraph_query(apiSubgraphs, idSubgraphs, graphqlQuery);
        if (!json_data_opt.has_value()) {
            return std::nullopt;
        }

        auto partial_data_opt = parse_pool_series(json_data_opt.value(), skip == 0);
        if (!partial_data_opt.has_value()) {
            return std::nullopt;
        }

        const auto& batch = partial_data_opt.value();

        // On first batch, set pool info fields and initialize Eigen matrices
        if (skip == 0) {
            full_data.token0_symbol = batch.token0_symbol;
            full_data.token1_symbol = batch.token1_symbol;
            full_data.fee_tier = batch.fee_tier;
            full_data.n_paths = 1;
            full_data.price = batch.price;
            full_data.volumeUSD = batch.volumeUSD;
            full_data.liquidity = batch.liquidity;
        } else {
            // Concatenate matrices horizontally (along columns)
            int prev_cols = static_cast<int>(full_data.price.cols());
            int new_cols = prev_cols + batch.price.cols();
            full_data.price.conservativeResize(1, new_cols);
            full_data.price.block(0, prev_cols, 1, batch.price.cols()) = batch.price;
            full_data.volumeUSD.conservativeResize(1, new_cols);
            full_data.volumeUSD.block(0, prev_cols, 1, batch.volumeUSD.cols()) = batch.volumeUSD;
            full_data.liquidity.conservativeResize(1, new_cols);
            full_data.liquidity.block(0, prev_cols, 1, batch.liquidity.cols()) = batch.liquidity;
        }

        // Append datetimes
        full_data.datetimes.insert(full_data.datetimes.end(), batch.datetimes.begin(), batch.datetimes.end());

        if (batch.datetimes.size() < static_cast<size_t>(batch_size)) {
            break; // no more results
        }

        skip += batch_size;

        if (skip > 10000) break; // safety cap 
    }

    full_data.n_points = static_cast<int>(full_data.datetimes.size());

    return full_data;
}


/**
 * @brief Write PoolSeries data to a tab-separated .txt file for easy import in Python or other environments.
 * @param filename Output file path.
 * @param pool PoolSeries object to write.
 */
void write_poolseries_to_txt(const std::string& filename, const PoolSeries& pool) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    // Write header
    ofs << "datetime\tprice\tvolumeUSD\tliquidity\n";
    for (int i = 0; i < pool.n_points; ++i) {
        ofs << pool.datetimes[i] << "\t"
            << pool.price(0, i) << "\t"
            << pool.volumeUSD(0, i) << "\t"
            << pool.liquidity(0, i) << "\n";
    }
    ofs.close();
}

std::optional<std::pair<std::string, std::string>> get_pool_token_names(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress) {
    // Use a minimal query to fetch only token symbols
    std::string query = R"({
    pool(id: ")" + poolAddress + R"(") {
        token0 { symbol }
        token1 { symbol }
    }
    })";
    auto response = run_subgraph_query(apiSubgraphs, idSubgraphs, query); // use graphql_post instead of run_subgraph_query
    if (!response) return std::nullopt;
    try {
        const auto& json_data = *response;
        if (json_data.contains("data") && json_data["data"].contains("pool")) {
            const auto& pool = json_data["data"]["pool"];
            std::string token0 = pool["token0"]["symbol"].get<std::string>();
            std::string token1 = pool["token1"]["symbol"].get<std::string>();
            return std::make_pair(token0, token1);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error fetching token names: " << e.what() << std::endl;
    }
    return std::nullopt;
}