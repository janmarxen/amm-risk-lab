#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "common.h" // for PriceSeries

/// Performs a GraphQL POST request and returns the parsed JSON or nullopt.
std::optional<nlohmann::json> run_subgraph_query(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& graphqlQuery
);

/// Parses a JSON response to extract price and datetime series.
std::optional<PriceSeries> parse_price_datetime_series(const nlohmann::json& json_data, int n_points = -1);

/// Fetches price and datetime series for a pool from the subgraph.
/// Returns std::nullopt on failure.
std::optional<PriceSeries> fetch_price_datetime_series(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress,
    const std::string& startDate,
    const std::string& endDate
);

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> load_pool_addresses(const std::string& json_path);

// Updated signature to require subgraph id
std::string get_pool_address(const std::string& symbol_pair, const std::string& subgraph_id);