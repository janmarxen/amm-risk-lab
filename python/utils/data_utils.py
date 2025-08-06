"""
data_utils.py

Modular utility functions for feature engineering and data preprocessing for AMM pool data.
All features that could leak information about future targets are properly shifted.
"""

import pandas as pd
import numpy as np

def remove_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """
    Remove outliers using IQR method with improved precision.
    
    Args:
        series: Input pandas Series
        k: IQR multiplier (default 3.0 for conservative outlier removal)
        
    Returns:
        Series with outliers replaced by NaN and filled
    """
    nonzero = series[series != 0]
    if len(nonzero) < 10:  # Need minimum samples for robust statistics
        return series
    
    # Use more precise percentile calculation
    q1 = nonzero.quantile(0.25, interpolation='linear')
    q3 = nonzero.quantile(0.75, interpolation='linear')
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    
    # Use more conservative replacement strategy
    filtered = series.where((series >= lower) & (series <= upper), np.nan)
    # Use forward-fill then backward-fill for better continuity
    filtered = filtered.fillna(method='ffill').fillna(method='bfill')
    return filtered

def calculate_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic return features."""
    df_copy = df.copy()
    
    # Basic returns
    df_copy['price_return'] = df_copy['price'].pct_change().astype(np.float64)
    df_copy['liquidity_return'] = df_copy['liquidity'].pct_change().astype(np.float64)
    df_copy['volume_return'] = df_copy['volumeUSD'].pct_change().astype(np.float64)
    
    # Apply outlier removal
    for col in ['price_return', 'liquidity_return', 'volume_return']:
        if col in df_copy.columns:
            df_copy[col] = remove_outliers_iqr(df_copy[col])
    
    return df_copy

def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility features with proper temporal alignment."""
    df_copy = df.copy()
    
    # Price volatility (no shift needed for price features as they don't leak target info)
    df_copy['price_volatility_3h'] = df_copy['price_return'].rolling(window=3).std()
    df_copy['price_volatility_6h'] = df_copy['price_return'].rolling(window=6).std()
    df_copy['price_volatility_24h'] = df_copy['price_return'].rolling(window=24).std()
    
    # Liquidity volatility (shifted by 1 to avoid leakage)
    df_copy['liquidity_volatility_3h'] = df_copy['liquidity_return'].shift(1).rolling(window=3).std()
    df_copy['liquidity_volatility_6h'] = df_copy['liquidity_return'].shift(1).rolling(window=6).std()
    df_copy['liquidity_volatility_24h'] = df_copy['liquidity_return'].shift(1).rolling(window=24).std()
    
    # Volume volatility (shifted by 1 to avoid leakage)
    df_copy['volume_volatility_3h'] = df_copy['volume_return'].shift(1).rolling(window=3).std()
    df_copy['volume_volatility_6h'] = df_copy['volume_return'].shift(1).rolling(window=6).std()
    df_copy['volume_volatility_24h'] = df_copy['volume_return'].shift(1).rolling(window=24).std()
    
    return df_copy

def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving average features with proper temporal alignment."""
    df_copy = df.copy()
    
    # Price moving averages (no shift needed for price features as they don't leak target info)
    df_copy['price_ma_3h'] = df_copy['price_return'].rolling(window=3).mean()
    df_copy['price_ma_6h'] = df_copy['price_return'].rolling(window=6).mean()
    df_copy['price_ma_24h'] = df_copy['price_return'].rolling(window=24).mean()
    
    # Liquidity moving averages (shifted by 1 to avoid leakage)
    df_copy['liquidity_ma_3h'] = df_copy['liquidity_return'].shift(1).rolling(window=3).mean()
    df_copy['liquidity_ma_6h'] = df_copy['liquidity_return'].shift(1).rolling(window=6).mean()
    df_copy['liquidity_ma_24h'] = df_copy['liquidity_return'].shift(1).rolling(window=24).mean()
    
    # Volume moving averages (shifted by 1 to avoid leakage)
    df_copy['volume_ma_3h'] = df_copy['volume_return'].shift(1).rolling(window=3).mean()
    df_copy['volume_ma_6h'] = df_copy['volume_return'].shift(1).rolling(window=6).mean()
    df_copy['volume_ma_24h'] = df_copy['volume_return'].shift(1).rolling(window=24).mean()
    
    return df_copy

def calculate_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-asset interaction features."""
    df_copy = df.copy()
    
    # Price-Volume correlations (mix current price with lagged volume/liquidity)
    df_copy['price_volume_corr_6h'] = df_copy['price_return'].rolling(6).corr(df_copy['volume_return'].shift(1))
    df_copy['price_volume_corr_24h'] = df_copy['price_return'].rolling(24).corr(df_copy['volume_return'].shift(1))
    df_copy['price_liquidity_corr_6h'] = df_copy['price_return'].rolling(6).corr(df_copy['liquidity_return'].shift(1))
    
    # Volume-Liquidity efficiency using returns (stationary)
    df_copy['volume_liquidity_return_ratio'] = (df_copy['volume_return'].shift(1) / 
                                               (df_copy['liquidity_return'].shift(1).abs() + 1e-8))
    df_copy['volume_liquidity_ratio_ma_6h'] = df_copy['volume_liquidity_return_ratio'].rolling(6).mean()
    df_copy['volume_liquidity_ratio_volatility'] = df_copy['volume_liquidity_return_ratio'].rolling(6).std()
    
    return df_copy

def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market microstructure features."""
    df_copy = df.copy()
    
    # Bid-ask spread proxies using returns (stationary)
    df_copy['price_volatility_to_volume_ratio'] = (df_copy['price_volatility_3h'] / 
                                                   (df_copy['volume_return'].shift(1).abs() + 1e-8))
    df_copy['liquidity_return_depth_proxy'] = (df_copy['liquidity_return'].shift(1).abs() / 
                                              (df_copy['price_volatility_3h'] + 1e-8))
    
    # Market impact indicators using returns (stationary)
    df_copy['market_impact_proxy'] = (df_copy['volume_return'].shift(1).abs() / 
                                     (df_copy['price_return'].abs() + 1e-8))
    df_copy['price_efficiency'] = (df_copy['price_return'].rolling(6).std() / 
                                  (df_copy['volume_return'].shift(1).rolling(6).std() + 1e-8))
    
    return df_copy

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and mean reversion features."""
    df_copy = df.copy()
    
    # Price momentum (no shift needed for price features)
    df_copy['price_momentum_3h'] = df_copy['price_return'].rolling(3).sum()
    df_copy['price_momentum_12h'] = df_copy['price_return'].rolling(12).sum()
    df_copy['price_momentum_24h'] = df_copy['price_return'].rolling(24).sum()
    
    # Liquidity momentum (shifted to avoid leakage)
    df_copy['liquidity_momentum_6h'] = df_copy['liquidity_return'].shift(1).rolling(6).sum()
    df_copy['liquidity_momentum_24h'] = df_copy['liquidity_return'].shift(1).rolling(24).sum()
    
    # Volume momentum (shifted to avoid leakage)
    df_copy['volume_momentum_6h'] = df_copy['volume_return'].shift(1).rolling(6).sum()
    df_copy['volume_momentum_24h'] = df_copy['volume_return'].shift(1).rolling(24).sum()
    
    # Mean reversion indicators (price features don't need shift)
    df_copy['price_deviation_from_ma_6h'] = df_copy['price_return'] - df_copy['price_ma_6h']
    df_copy['price_deviation_from_ma_24h'] = df_copy['price_return'] - df_copy['price_ma_24h']
    df_copy['liquidity_deviation_from_ma_6h'] = df_copy['liquidity_return'].shift(1) - df_copy['liquidity_ma_6h']
    
    return df_copy

def calculate_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility regime features."""
    df_copy = df.copy()
    
    # Volatility regimes (using historical data to avoid leakage)
    df_copy['high_vol_regime_3h'] = (df_copy['price_volatility_3h'] > 
                                    df_copy['price_volatility_3h'].shift(1).rolling(168).quantile(0.8)).astype(int)
    df_copy['high_vol_regime_24h'] = (df_copy['price_volatility_24h'] > 
                                     df_copy['price_volatility_24h'].shift(1).rolling(168).quantile(0.8)).astype(int)
    
    # Volatility clustering (shifted to avoid leakage)
    vol_mean_24h = df_copy['price_volatility_3h'].shift(1).rolling(24).mean()
    df_copy['vol_cluster_indicator'] = ((df_copy['price_volatility_3h'] > vol_mean_24h) & 
                                       (df_copy['price_volatility_3h'].shift(1) > vol_mean_24h.shift(1))).astype(int)
    
    # Volatility term structure
    df_copy['vol_term_structure'] = df_copy['price_volatility_3h'] / (df_copy['price_volatility_24h'] + 1e-8)
    
    return df_copy

def calculate_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate liquidity-specific features using returns (stationary)."""
    df_copy = df.copy()
    
    # Liquidity concentration using returns (stationary)
    liquidity_return_rolling_max = df_copy['liquidity_return'].shift(1).rolling(168).max()
    df_copy['liquidity_return_concentration'] = (df_copy['liquidity_return'].shift(1) / 
                                                (liquidity_return_rolling_max + 1e-8))  # Weekly max ratio
    df_copy['liquidity_return_percentile_6h'] = df_copy['liquidity_return'].shift(1).rolling(6).rank(pct=True)
    df_copy['liquidity_return_percentile_24h'] = df_copy['liquidity_return'].shift(1).rolling(24).rank(pct=True)
    
    # Liquidity stress indicators using returns (stationary)
    df_copy['liquidity_stress'] = (df_copy['liquidity_return'].shift(1) < 
                                  df_copy['liquidity_return'].shift(1).rolling(168).quantile(0.1)).astype(int)
    df_copy['liquidity_abundance'] = (df_copy['liquidity_return'].shift(1) > 
                                     df_copy['liquidity_return'].shift(1).rolling(168).quantile(0.9)).astype(int)
    
    # Liquidity stability using volatility (stationary)
    df_copy['liquidity_stability'] = 1 / (df_copy['liquidity_volatility_6h'] + 1e-8)
    
    return df_copy

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based cyclical features."""
    df_copy = df.copy()
    
    # Basic temporal features
    df_copy['hour'] = df_copy['datetime'].dt.hour
    df_copy['day_of_week'] = df_copy['datetime'].dt.dayofweek
    df_copy['month'] = df_copy['datetime'].dt.month
    
    # Cyclical encodings
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
    
    # Market session indicators
    df_copy['us_market_hours'] = ((df_copy['hour'] >= 14) & (df_copy['hour'] <= 21)).astype(int)  # 9 AM - 4 PM EST in UTC
    df_copy['asia_market_hours'] = ((df_copy['hour'] >= 0) & (df_copy['hour'] <= 8)).astype(int)  # Asia trading hours
    df_copy['weekend'] = (df_copy['day_of_week'].isin([5, 6])).astype(int)
    
    # Season encoding
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    df_copy['season'] = df_copy['month'].apply(get_season)
    
    return df_copy

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced technical indicators."""
    df_copy = df.copy()
    
    # RSI-like indicators
    def calculate_rsi(returns, window=14):
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gains = gains.rolling(window).mean()
        avg_losses = losses.rolling(window).mean()
        rs = avg_gains / (avg_losses + 1e-8)
        return 100 - (100 / (1 + rs))
    
    # Price RSI (no shift needed for price features)
    df_copy['price_rsi_6h'] = calculate_rsi(df_copy['price_return'], 6)
    df_copy['price_rsi_24h'] = calculate_rsi(df_copy['price_return'], 24)
    
    # Liquidity RSI (shifted to avoid leakage)
    df_copy['liquidity_rsi_6h'] = calculate_rsi(df_copy['liquidity_return'].shift(1), 6)
    
    # Volume RSI (shifted to avoid leakage)
    df_copy['volume_rsi_6h'] = calculate_rsi(df_copy['volume_return'].shift(1), 6)
    
    # Bollinger Band-like features (use current price with lagged moving averages)
    df_copy['price_bb_upper_6h'] = df_copy['price_ma_6h'] + 2 * df_copy['price_volatility_6h']
    df_copy['price_bb_lower_6h'] = df_copy['price_ma_6h'] - 2 * df_copy['price_volatility_6h']
    df_copy['price_bb_position'] = ((df_copy['price_return'] - df_copy['price_bb_lower_6h']) / 
                                   (df_copy['price_bb_upper_6h'] - df_copy['price_bb_lower_6h'] + 1e-8))
    
    return df_copy

def calculate_shock_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate shock and event detection features."""
    df_copy = df.copy()
    
    # Price shocks (use current price returns with current volatility)
    df_copy['price_shock_3h'] = (df_copy['price_return'].abs() > 
                                3 * df_copy['price_volatility_3h']).astype(int)
    df_copy['price_shock_6h'] = (df_copy['price_return'].abs() > 
                                3 * df_copy['price_volatility_6h']).astype(int)
    
    # Volume spikes (shifted to avoid leakage)
    df_copy['volume_spike'] = (df_copy['volume_return'].shift(1) > 
                              df_copy['volume_return'].shift(1).rolling(24).quantile(0.95)).astype(int)
    df_copy['volume_drought'] = (df_copy['volume_return'].shift(1) < 
                                df_copy['volume_return'].shift(1).rolling(24).quantile(0.05)).astype(int)
    
    # Liquidity events (shifted to avoid leakage)
    df_copy['liquidity_drain'] = (df_copy['liquidity_return'].shift(1) < -0.1).astype(int)  # 10% liquidity drop
    df_copy['liquidity_injection'] = (df_copy['liquidity_return'].shift(1) > 0.1).astype(int)  # 10% liquidity increase
    
    return df_copy

