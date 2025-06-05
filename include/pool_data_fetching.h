#pragma once
#include <string>
#include <optional>
#include <nlohmann/json.hpp>
#include "common.h"

/**
 * @brief Fetches pool data (metadata and hourly stats) from a subgraph->poolAddress.
 * 
 * @param apiSubgraphs   The API key/token for The Graph subgraphs gateway.
 * @param idSubgraphs    The subgraph ID to query.
 * @param poolAddress    The pool contract address (lowercase hex string).
 * @param startDate      Start date in "YYYY-MM-DD" format (UTC).
 * @param endDate        End date in "YYYY-MM-DD" format (UTC).
 * @return std::optional<PoolData> 
 *         Returns PoolData struct with metadata and time series if successful, std::nullopt otherwise.
 */
std::optional<PoolData> fetch_pool_data(const std::string& apiSubgraphs, 
    const std::string& idSubgraphs, 
    const std::string& poolAddress, 
    const std::string& startDate, 
    const std::string& endDate);

