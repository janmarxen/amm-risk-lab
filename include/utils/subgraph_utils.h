#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

/// Helper struct for price and datetime series.
struct PriceDatetimeSeries {
    std::vector<std::string> datetime;
    std::vector<double> price;
};

/// Performs a GraphQL POST request and returns the parsed JSON or nullopt.
std::optional<nlohmann::json> graphql_post(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& graphqlQuery
);

/// Parses a JSON response to extract price and datetime series.
std::optional<PriceDatetimeSeries> parse_price_datetime_series(const nlohmann::json& json_data);

/// Fetches price and datetime series for a pool from the subgraph.
/// Returns std::nullopt on failure.
std::optional<PriceDatetimeSeries> fetch_price_datetime_series(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress,
    const std::string& startDate,
    const std::string& endDate
);

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> load_pool_addresses(const std::string& json_path);

// Updated signature to require subgraph id
std::string get_pool_address(const std::string& symbol_pair, const std::string& subgraph_id);