#include "common.h"
#include <Eigen/Dense>
#include <string>

/**
 * @brief Dummy prediction for the PLV model.
 *        Fills liquidity and volumeUSD with constant dummy values.
 * @param price_series The input price series (multiple paths supported).
 * @param token0_symbol Symbol for token0.
 * @param token1_symbol Symbol for token1.
 * @param fee_tier Fee tier for the pool.
 * @return PoolUSDSeries with dummy liquidity and volumeUSD.
 */
PoolSeries dummy_predict_plv(
    const PriceSeries& price_series,
    const std::string& token0_symbol,
    const std::string& token1_symbol,
    double fee_tier
) {
    int n_paths = price_series.n_paths;
    int n_points = price_series.n_points;

    // Dummy values
    double dummy_liquidity = 1e6;
    double dummy_volume = 1e4;

    Eigen::MatrixXd liquidity = Eigen::MatrixXd::Constant(n_paths, n_points, dummy_liquidity);
    Eigen::MatrixXd volumeUSD = Eigen::MatrixXd::Constant(n_paths, n_points, dummy_volume);

    return PoolSeries(
        price_series.datetimes,
        token0_symbol,
        token1_symbol,
        fee_tier,
        price_series.prices,
        volumeUSD,
        liquidity
    );
}
