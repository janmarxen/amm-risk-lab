#include "utils/conversions.h"
#include "utils/time_utils.h"
#include "utils/subgraph_utils.h"
#include "common.h"
#include <unordered_map>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <optional>
#include <algorithm>

// Helper to fetch price series and token symbols for a pool
static std::optional<PoolPriceSeries> fetch_pool_price_series(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress,
    const std::string& startDate,
    const std::string& endDate
) {
    time_t start_ts = date_to_unix(startDate);
    time_t end_ts = date_to_unix(endDate) + 86399;

    std::string graphqlQuery = R"({
    pool(id: ")" + poolAddress + R"(") {
        token0 { symbol }
        token1 { symbol }
    }
    poolHourDatas(
        orderBy: periodStartUnix,
        orderDirection: asc,
        where: { pool: ")" + poolAddress + R"(", periodStartUnix_gte: )" + std::to_string(start_ts) + R"(, periodStartUnix_lte: )" + std::to_string(end_ts) + R"( }
    ) {
        periodStartUnix
        token0Price
    }
    })";

    auto json_data_opt = run_subgraph_query(apiSubgraphs, idSubgraphs, graphqlQuery);
    if (!json_data_opt.has_value()) return std::nullopt;
    const auto& json_data = json_data_opt.value();

    std::string token0_symbol = "";
    std::string token1_symbol = "";
    std::vector<std::string> datetimes;
    std::vector<double> prices;

    if (json_data.contains("data") && json_data["data"].contains("pool")) {
        const auto& pool = json_data["data"]["pool"];
        token0_symbol = pool["token0"]["symbol"].get<std::string>();
        token1_symbol = pool["token1"]["symbol"].get<std::string>();
    }
    if (json_data.contains("data") && json_data["data"].contains("poolHourDatas")) {
        for (const auto& hour_data : json_data["data"]["poolHourDatas"]) {
            time_t timestamp = hour_data["periodStartUnix"].get<time_t>();
            datetimes.push_back(unix_to_datetime(timestamp));
            prices.push_back(std::stod(hour_data["token0Price"].get<std::string>()));
        }
    }
    int n_points = static_cast<int>(datetimes.size());
    Eigen::MatrixXd price_mat(1, n_points);
    for (int i = 0; i < n_points; ++i) price_mat(0, i) = prices[i];
    PriceSeries price_series(datetimes, price_mat);

    PoolPriceSeries series(token0_symbol, token1_symbol, price_series);
    return series;
}

// Main conversion function
std::vector<double> convert_to_usd(
    const std::vector<double>& values,
    const std::string& currency_symbol,
    const std::vector<std::string>& datetime,
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs
) {
    if (currency_symbol == "USD") return values;

    // Try to find a pool address for currency_symbol/USDC or USDC/currency_symbol
    std::string symbol_pair = currency_symbol + "_USDC";
    std::string pool_address = get_pool_address(symbol_pair, idSubgraphs);
    bool currency_is_token0 = true;
    if (pool_address.empty()) {
        symbol_pair = "USDC_" + currency_symbol;
        pool_address = get_pool_address(symbol_pair, idSubgraphs);
        currency_is_token0 = false;
    }

    if (pool_address.empty()) {
        std::cerr << "No pool address found for " << currency_symbol << "/USDC" << std::endl;
        return values;
    }

    std::string startDate = datetime.empty() ? "" : datetime.front().substr(0, 10);
    std::string endDate = datetime.empty() ? "" : datetime.back().substr(0, 10);

    // Fetch price series and token symbols for the pool
    auto price_info_opt = fetch_pool_price_series(apiSubgraphs, idSubgraphs, pool_address, startDate, endDate);
    if (!price_info_opt.has_value()) {
        std::cerr << "Failed to fetch price info for pool " << pool_address << std::endl;
        return values;
    }
    const auto& price_info = price_info_opt.value();

    // Determine which token is USD (USDC, USDT, DAI, etc.)
    std::string usd_token = "";
    std::vector<std::string> usd_like = {"USDC", "USDT", "DAI", "USD"};
    for (const auto& s : usd_like) {
        if (strcasecmp(price_info.token0_symbol.c_str(), s.c_str()) == 0) usd_token = price_info.token0_symbol;
        if (strcasecmp(price_info.token1_symbol.c_str(), s.c_str()) == 0) usd_token = price_info.token1_symbol;
    }
    if (usd_token.empty()) {
        std::cerr << "No USD-like token found in pool " << pool_address << std::endl;
        return values;
    }

    // Map datetime to price for fast lookup
    std::unordered_map<std::string, double> dt_to_price;
    for (int i = 0; i < price_info.price_series.n_points; ++i) {
        dt_to_price[price_info.price_series.datetimes[i]] = price_info.price_series.prices(0, i);
    }

    // Conversion logic:
    // token0Price = token0 per token1
    // If currency_symbol == token0 and token1 is USD: USD = value * price
    // If currency_symbol == token1 and token0 is USD: USD = value / price
    // Otherwise, fallback to original logic (may be wrong for non-USD pools)
    std::vector<double> usd_values;
    usd_values.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        double price = 1.0;
        auto it = dt_to_price.find(datetime[i]);
        if (it != dt_to_price.end()) price = it->second;

        double usd_val = values[i];
        if (strcasecmp(price_info.token0_symbol.c_str(), currency_symbol.c_str()) == 0 &&
            std::find_if(usd_like.begin(), usd_like.end(), [&](const std::string& s) {
                return strcasecmp(price_info.token1_symbol.c_str(), s.c_str()) == 0;
            }) != usd_like.end()) {
            // currency is token0, token1 is USD: USD = value / price
            usd_val = values[i] / price;
        } else if (strcasecmp(price_info.token1_symbol.c_str(), currency_symbol.c_str()) == 0 &&
                   std::find_if(usd_like.begin(), usd_like.end(), [&](const std::string& s) {
                       return strcasecmp(price_info.token0_symbol.c_str(), s.c_str()) == 0;
                   }) != usd_like.end()) {
            // currency is token1, token0 is USD: USD = value * price
            usd_val = values[i] * price;
        } else {
            // fallback: try to use price as multiplier
            usd_val = values[i] / price;
        }
        usd_values.push_back(usd_val);
    }
    return usd_values;
}



