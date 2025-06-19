#pragma once

#include "common.h"
#include <string>

/**
 * @brief Dummy prediction for the PLV model.
 *        Fills liquidity and volumeUSD with constant dummy values.
 * @param price_series The input price series (multiple paths supported).
 * @param token0_symbol Symbol for token0.
 * @param token1_symbol Symbol for token1.
 * @param fee_tier Fee tier for the pool.
 * @return PoolSeries with dummy liquidity and volumeUSD.
 */
PoolSeries dummy_predict_plv(
    const PriceSeries& price_series,
    const std::string& token0_symbol,
    const std::string& token1_symbol,
    double fee_tier
);
