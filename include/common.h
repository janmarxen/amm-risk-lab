#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

/// Represents aggregated data for a liquidity pool fetched from the subgraph.
struct PoolData {
    std::string token0_symbol; // Symbol of the first token in the pool (token0.symbol).
    std::string token1_symbol; // Symbol of the second token in the pool (token1.symbol).
    double fee_tier; // Fee tier of the pool (feeTier), e.g., 0.003 for 0.3% fee.
    size_t data_points; // Number of data points retrieved for this pool.
    /// Vector of timestamps (as datetime strings) corresponding to each data point.
    /// These correspond to poolHourDatas.periodStartUnix converted to readable date-time.
    std::vector<std::string> datetime;
    std::vector<double> price; // Vector of token0 prices (in terms of token1) at each timestamp (poolHourDatas.token0Price).
    // std::vector<double> volumeUSD; // Vector of volume in USD at each timestamp (poolHourDatas.volumeUSD).
    std::vector<double> volume0; // Vector of token0 volume at each timestamp (poolHourDatas.volume0).      
    std::vector<double> volume1; // Vector of token1 volume at each timestamp (poolHourDatas.volume1).
    std::vector<double> volumeUSD; // Vector of tokens volume in USD at each timestamp (poolHourDatas.volumeUSD).
    std::vector<double> liquidity; // Vector of global liquidity values at each timestamp (poolHourDatas.liquidity).
};

// Holds a time series of prices and their corresponding datetimes (ISO 8601 strings)
struct PriceSeries {
    std::vector<std::string> datetimes; // ISO 8601 datetime strings
    Eigen::MatrixXd prices; // n_paths x N matrix (N = number of steps or 1 if final_only)
};
