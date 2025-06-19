#pragma once

#include "common.h"
#include "lp_simulation.h"
#include <string>
#include <optional>
#include <nlohmann/json.hpp>

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
std::optional<PoolSeries> fetch_pool_series(const std::string& apiSubgraphs, 
    const std::string& idSubgraphs, 
    const std::string& poolAddress, 
    const std::string& startDate, 
    const std::string& endDate);

/**
 * @brief Write PoolSeries data to a tab-separated .txt file for easy import in Python or other environments.
 * @param filename Output file path.
 * @param pool PoolSeries object to write.
 */
void write_poolseries_to_txt(const std::string& filename, const PoolSeries& pool);

/**
 * @brief Fetches the token0 and token1 symbols for a given pool address.
 * @param apiSubgraphs   The API key/token for The Graph subgraphs gateway.
 * @param idSubgraphs    The subgraph ID to query.
 * @param poolAddress    The pool contract address (lowercase hex string).
 * @return std::optional<std::pair<std::string, std::string>>
 *         Returns a pair of token0 and token1 symbols if successful, std::nullopt otherwise.
 */
std::optional<std::pair<std::string, std::string>> get_pool_token_names(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress);

/**
 * @brief Fetch the actual LP position data from the subgraph API and fill LPSimulationResults.
 * @param apiSubgraphs   The API key/token for The Graph subgraphs gateway.
 * @param idSubgraphs    The subgraph ID to query.
 * @param poolAddress    The pool contract address (lowercase hex string).
 * @param ownerAddress   The LP's wallet address (lowercase hex string).
 * @param startDate      Start date in "YYYY-MM-DD" format (UTC).
 * @param endDate        End date in "YYYY-MM-DD" format (UTC).
 * @return LPSimulationResults Actual results for the LP position.
 */
LPSimulationResults fetch_actual_lp_position(
    const std::string& apiSubgraphs,
    const std::string& idSubgraphs,
    const std::string& poolAddress,
    const std::string& ownerAddress,
    const std::string& startDate,
    const std::string& endDate
);


