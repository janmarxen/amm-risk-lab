#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

/// Base class for any time series with datetimes
class TimeSeries {
  public:
    std::vector<std::string> datetimes; // ISO 8601 datetime strings
    int n_points;

    TimeSeries() : n_points(0) {}
    TimeSeries(const std::vector<std::string>& datetimes_)
        : datetimes(datetimes_), n_points(static_cast<int>(datetimes_.size())) {}
    virtual ~TimeSeries() = default;
};

/// Holds a time series of prices and their corresponding datetimes (ISO 8601 strings)
class PriceSeries : public TimeSeries {
  public:
    int n_paths; // Number of paths (n_paths)
    Eigen::MatrixXd prices; // n_paths x N matrix (N = number of steps or 1 if final_only)

    PriceSeries() : TimeSeries(), n_paths(0), prices() {}
    PriceSeries(const std::vector<std::string>& datetimes_, const Eigen::MatrixXd& prices_)
        : TimeSeries(datetimes_), prices(prices_) {
        n_points = prices_.cols();
        n_paths = prices_.rows();
    }
};

/// Class to hold price series and token symbols for a pool
class PoolPriceSeries {
  public:
    std::string token0_symbol;
    std::string token1_symbol;
    PriceSeries price_series;

    PoolPriceSeries() : token0_symbol(""), token1_symbol(""), price_series() {}
    PoolPriceSeries(const std::string& t0, const std::string& t1, const PriceSeries& ps)
        : token0_symbol(t0), token1_symbol(t1), price_series(ps) {}
};

/// Represents aggregated data for a liquidity pool fetched from the subgraph (USD only).
class PoolSeries : public TimeSeries {
  public:
    std::string token0_symbol;
    std::string token1_symbol;
    double fee_tier;
    int n_paths;
    Eigen::MatrixXd price;      // n_paths x N matrix of token0 prices (in terms of token1)
    Eigen::MatrixXd volumeUSD;  // n_paths x N matrix of tokens volume in USD
    Eigen::MatrixXd liquidity;  // n_paths x N matrix of global liquidity values

    PoolSeries()
        : TimeSeries(), token0_symbol(""), token1_symbol(""), fee_tier(0.0), n_paths(0),
          price(), volumeUSD(), liquidity() {}

    PoolSeries(
        const std::vector<std::string>& datetimes_,
        const std::string& t0,
        const std::string& t1,
        double fee,
        const Eigen::MatrixXd& price_,
        const Eigen::MatrixXd& volumeUSD_,
        const Eigen::MatrixXd& liquidity_
    )
        : TimeSeries(datetimes_), token0_symbol(t0), token1_symbol(t1), fee_tier(fee),
          n_paths(price_.rows()), price(price_), volumeUSD(volumeUSD_), liquidity(liquidity_)
    {
        n_points = price_.cols();
    }
};


