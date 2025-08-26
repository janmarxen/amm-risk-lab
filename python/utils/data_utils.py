"""
data_utils.py

Utility functions for feature engineering and data preprocessing for AMM pool data.

This module provides a complete feature engineering pipeline for Automated Market Maker (AMM) 
pool data, specifically designed for liquidity provider (LP) performance prediction tasks.
All features are carefully designed to avoid data leakage while capturing market dynamics.

Key Features:
- 70+ engineered features across 10 categories
- Intelligent missing value handling with feature-type-specific strategies  
- Temporal alignment to prevent future information leakage
- Robust outlier detection and handling
- Multi-task learning support for zero-inflated targets

Feature Categories:
1. Basic Returns: Price, liquidity, volume percentage changes
2. Volatility: Rolling standard deviations across multiple horizons
3. Moving Averages: Trend indicators and smoothed signals
4. Cross-Asset: Correlations and interaction effects
5. Microstructure: Market quality and trading cost proxies
6. Momentum: Trend persistence and mean reversion signals
7. Volatility Regimes: Market stress and clustering detection
8. Liquidity: Supply dynamics and stress indicators
9. Temporal: Time-based cyclical patterns and market sessions
10. Technical: RSI, Bollinger bands, and other TA indicators
11. Shock Detection: Extreme event and anomaly identification

Data Leakage Prevention:
- Price features: No temporal shift (exogenous variable)
- Liquidity features: 1-period lag (endogenous target)
- Volume features: 1-period lag (endogenous target)

Usage:
    from python.utils.data_utils import feature_engineer, get_X_y
    
    # Engineer features for a pool
    df_engineered = feature_engineer(df_raw, verbose=True)
    
    # Prepare for hybrid multi-task learning
    X, y_cls_1, y_reg_1, y_reg_2 = get_X_y(
        df_engineered, features, target_cols, n_lags
    )
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings

def feature_engineer(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add engineered features to a pool DataFrame, including returns, rolling stats, and temporal features.
    All features that could leak information about future targets are properly shifted.
    
    Args:
        df (pd.DataFrame): Raw pool data with columns ['price', 'liquidity', 'volumeUSD', 'periodStartUnix']
        verbose (bool): Whether to print feature engineering summary information
        
    Returns:
        pd.DataFrame: DataFrame with 70+ engineered features added, missing values handled intelligently
    """
    # Suppress numpy warnings about invalid values during feature calculation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['periodStartUnix'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Ensure proper data types
        df['price'] = df['price'].astype(np.float64)
        df['liquidity'] = df['liquidity'].astype(np.float64)
        df['volumeUSD'] = df['volumeUSD'].astype(np.float64)
    
        # Apply modular feature engineering functions
        df = calculate_basic_returns(df)
        df = calculate_volatility_features(df)
        df = calculate_moving_averages(df)
        df = calculate_cross_asset_features(df)
        df = calculate_microstructure_features(df)
        df = calculate_momentum_features(df)
        df = calculate_volatility_regime_features(df)
        df = calculate_liquidity_features(df)
        df = calculate_temporal_features(df)
        df = calculate_technical_indicators(df)
        df = calculate_shock_detection_features(df)
        
        # Handle missing values intelligently based on feature types
        df = df.replace([np.inf, -np.inf], np.nan)
        df = handle_missing_values(df, verbose=False)
        
        # # Final cleanup: Replace any remaining infinite values with NaN
        # # This catches any infinities that might have been introduced by complex calculations
        # df = df.replace([np.inf, -np.inf], np.nan)
        
        # # Fill any new NaN values created by infinity replacement
        # numeric_cols = df.select_dtypes(include=[np.number]).columns
        # for col in numeric_cols:
        #     if df[col].isnull().any():
        #         df[col] = df[col].ffill().bfill().fillna(0)
        
        # Print feature engineering summary
        if verbose:
            final_rows = len(df)
            nan_rows = df.isnull().any(axis=1).sum()
            nan_cols = df.columns[df.isnull().any()].tolist()
            total_features = len(df.columns) - 1  # Exclude datetime
            
            print(f"Feature engineering complete:")
            print(f"  Initial rows: {initial_rows}")
            print(f"  Final rows: {final_rows}")
            print(f"  Total features: {total_features}")
            print(f"  Rows with NaNs: {nan_rows}/{final_rows} ({nan_rows/final_rows*100:.2f}%)")
            if nan_rows > 0:
                print(f"  Columns with NaNs: {len(nan_cols)} ({', '.join(nan_cols[:5])}{'...' if len(nan_cols) > 5 else ''})")
        
        # Clean up
        if 'periodStartUnix' in df.columns:
            df = df.drop(columns=['periodStartUnix'])
        
        return df

def get_X_y(df: pd.DataFrame, features: List[str], target_cols: List[str], n_lags: int) -> Tuple[list, list, list, list]:
    """
    Convert a DataFrame to supervised learning arrays for Hybrid Multi-task model.
    
    Creates time series sequences where:
    - X: lagged features (ending at t) + target lags (ending at t-1)
    - y_cls_1: classification labels (1 if target == 0, else 0) for Task 1 (liquidity_return) only
    - y_reg_1, y_reg_2: regression targets (at time t) for both tasks
    
    Task 1 (liquidity_return): Zero-inflated (classification + regression)
    Task 2 (volume_return): Standard regression only
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and targets
        features (List[str]): List of feature column names
        target_cols (List[str]): List of exactly 2 target column names [target1, target2]
        n_lags (int): Number of lag steps for sequence creation
        
    Returns:
        Tuple[list, list, list, list]: (X, y_cls_1, y_reg_1, y_reg_2)
        - X: Feature sequences of shape (N, n_lags, num_features + 2)
        - y_cls_1: Binary classification targets for Task 1 only (zero-inflation indicators)
        - y_reg_1, y_reg_2: Continuous regression targets for both tasks
        
    Raises:
        ValueError: If target_cols doesn't contain exactly 2 targets
    """
    if len(target_cols) != 2:
        raise ValueError("Multi-task model requires exactly 2 target columns")
    
    target_col_1, target_col_2 = target_cols
    
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + target_cols)

    data_feats = df[features].to_numpy()
    data_target_1 = df[target_col_1].to_numpy()
    data_target_2 = df[target_col_2].to_numpy()
    T = len(df)

    if T < n_lags + 1:
        return [], [], [], []

    # Feature window: t - n_lags + 1 to t (length = n_lags, INCLUDE present)
    feats_window = np.lib.stride_tricks.sliding_window_view(data_feats, (n_lags, data_feats.shape[1]))
    feats_window = feats_window[:, 0, :, :]  # shape: (T - n_lags + 1, n_lags, num_features)

    # Target lag windows for both targets: t - n_lags to t - 1 (length = n_lags, EXCLUDE present)
    target_1_lags = np.lib.stride_tricks.sliding_window_view(data_target_1, n_lags + 1)
    target_1_lags = target_1_lags[:, :-1]  # Remove value at t
    target_1_lags = target_1_lags[:, :, np.newaxis]  # shape: (N, n_lags, 1)
    
    target_2_lags = np.lib.stride_tricks.sliding_window_view(data_target_2, n_lags + 1)
    target_2_lags = target_2_lags[:, :-1]  # Remove value at t
    target_2_lags = target_2_lags[:, :, np.newaxis]  # shape: (N, n_lags, 1)

    # Targets at time t
    y_1 = data_target_1[n_lags:]
    y_2 = data_target_2[n_lags:]

    # Match lengths
    min_len = min(len(feats_window), len(target_1_lags), len(target_2_lags), len(y_1), len(y_2))
    feats_window = feats_window[-min_len:]
    target_1_lags = target_1_lags[-min_len:]
    target_2_lags = target_2_lags[-min_len:]
    y_1 = y_1[-min_len:]
    y_2 = y_2[-min_len:]

    # Concatenate features and both target lags
    X = np.concatenate([feats_window, target_1_lags, target_2_lags], axis=2)  # shape: (N, n_lags, num_features + 2)

    # Filter valid samples
    mask = (np.isfinite(X).all(axis=(1, 2)) & 
            np.isfinite(y_1) & np.isfinite(y_2))
    X = X[mask]
    y_reg_1 = y_1[mask]
    y_reg_2 = y_2[mask]
    # Task 1 (liquidity_return): Zero-inflated - create classification targets
    y_cls_1 = (y_reg_1 == 0).astype(float)
    # Task 2 (volume_return): Standard regression - no classification needed

    return X.tolist(), y_cls_1.tolist(), y_reg_1.tolist(), y_reg_2.tolist()

def dropna_features_targets(df: pd.DataFrame, features: List[str], target_cols) -> pd.DataFrame:
    """
    Drop rows with NaNs in any of the selected features or target columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature column names to check for NaNs
        target_cols (Union[str, List[str]]): Target column name (str) or list of target column names
        
    Returns:
        pd.DataFrame: DataFrame with rows containing NaNs in specified columns removed,
                     index reset to maintain sequential order
    """
    # Handle both single target (str) and multiple targets (list)
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    mask = df[features + target_cols].notnull().all(axis=1)
    return df.loc[mask].reset_index(drop=True)

def handle_missing_values(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Intelligently handle missing values in engineered features based on feature type.
    Uses feature-specific strategies for optimal data preservation and minimal bias.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features
        verbose (bool): Whether to print debug information for problematic columns
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled using intelligent strategies:
        - Returns: forward/backward fill → 0 (no change)
        - Correlations: forward/backward fill → median (if available) → 0
        - RSI indicators: forward/backward fill → 50 (neutral)
        - Volatility: forward/backward fill → median → 0.01
        - Percentiles: forward/backward fill → 0.5 (median rank)
        - Binary indicators: forward/backward fill → 0 (no event)
        - Time features: forward/backward fill only
        - Others: forward/backward fill → median → 0
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    # # Debug: Check specific problematic columns before interpolation
    # if verbose and 'volume_return' in df_copy.columns:
    #     vol_ret_nans = df_copy['volume_return'].isnull().sum()
    #     vol_ret_total = len(df_copy['volume_return'])
    #     if vol_ret_nans == vol_ret_total:
    #         print(f"  WARNING: All volume_return values are NaN - likely constant volumeUSD")
    #         unique_volumes = df_copy['volumeUSD'].nunique() if 'volumeUSD' in df_copy.columns else 'N/A'
    #         print(f"  Unique volumeUSD values: {unique_volumes}")
    
    for col in numeric_cols:
        if col not in ['datetime']:  # Don't interpolate datetime
            # Apply intelligent filling based on feature type
            if col.endswith('_return') or col.startswith('price_return') or col.startswith('liquidity_return') or col.startswith('volume_return'):
                # For returns: forward fill, backward fill, then fill with 0 (meaning no change)
                df_copy[col] = df_copy[col].ffill().bfill().fillna(0)
            elif 'correlation' in col or 'corr' in col:
                # For correlations: try interpolation, then fill with median correlation if available, else 0
                df_copy[col] = df_copy[col].ffill().bfill()
                if df_copy[col].isnull().any():
                    # Use median of available correlations, fallback to 0
                    median_corr = df_copy[col].median() if df_copy[col].notna().sum() > 0 else 0.0
                    df_copy[col] = df_copy[col].fillna(median_corr)
            elif col.endswith('_rsi_6h') or col.endswith('_rsi_24h'):
                # For RSI: interpolate, then fill with neutral value (50)
                df_copy[col] = df_copy[col].ffill().bfill().fillna(50.0)
            elif 'volatility' in col:
                # For volatility: interpolate, then fill with median or small positive value
                df_copy[col] = df_copy[col].ffill().bfill()
                if df_copy[col].isnull().any():
                    median_vol = df_copy[col].median() if df_copy[col].notna().sum() > 0 else 0.01
                    df_copy[col] = df_copy[col].fillna(median_vol)
            elif col.endswith('_percentile_6h') or col.endswith('_percentile_24h'):
                # For percentiles: interpolate, then fill with 0.5 (median rank)
                df_copy[col] = df_copy[col].ffill().bfill().fillna(0.5)
            elif col in ['hour', 'day_of_week', 'month', 'season']:
                # For time features: forward/backward fill only (these shouldn't have NaNs anyway)
                df_copy[col] = df_copy[col].ffill().bfill()
            elif col.endswith('_sin') or col.endswith('_cos'):
                # For cyclical features: forward/backward fill only
                df_copy[col] = df_copy[col].ffill().bfill()
            elif col.endswith('_indicator') or col.endswith('_regime') or 'shock' in col or 'spike' in col or 'stress' in col:
                # For binary indicators: interpolate, then fill with 0 (no event)
                df_copy[col] = df_copy[col].ffill().bfill().fillna(0)
            else:
                # For other features: interpolate, then fill with median if available, else 0
                df_copy[col] = df_copy[col].ffill().bfill()
                if df_copy[col].isnull().any():
                    median_val = df_copy[col].median() if df_copy[col].notna().sum() > 0 else 0.0
                    df_copy[col] = df_copy[col].fillna(median_val)
    
    return df_copy

def remove_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """
    Remove outliers using the Interquartile Range (IQR) method with improved precision.
    
    Identifies outliers as values outside [Q1 - k*IQR, Q3 + k*IQR] range, where
    Q1 and Q3 are the 25th and 75th percentiles. Outliers are replaced with NaN
    and then filled using forward/backward fill for temporal continuity.
    
    Args:
        series (pd.Series): Input pandas Series to process
        k (float): IQR multiplier for outlier detection. Default 3.0 for conservative removal.
                  Higher values = fewer outliers removed, lower values = more aggressive removal.
        
    Returns:
        pd.Series: Series with outliers replaced and filled. Original series returned
                  if insufficient data (<10 non-zero values) for robust statistics.
                  
    Note:
        - Only considers non-zero, finite values for percentile calculation to avoid bias
        - Uses linear interpolation for more precise percentile calculation
        - Applies forward-fill then backward-fill for temporal continuity in time series
    """
    # First replace infinities with NaN
    series_clean = series.replace([np.inf, -np.inf], np.nan)
    
    # Filter to non-zero, finite values for robust statistics
    nonzero_finite = series_clean[(series_clean != 0) & (series_clean.notna())]
    if len(nonzero_finite) < 10:  # Need minimum samples for robust statistics
        # Still clean infinities even if we can't do IQR
        return series_clean.ffill().bfill().fillna(0)
    
    # Use more precise percentile calculation on finite values
    q1 = nonzero_finite.quantile(0.25, interpolation='linear')
    q3 = nonzero_finite.quantile(0.75, interpolation='linear')
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    
    # Use more conservative replacement strategy
    filtered = series_clean.where((series_clean >= lower) & (series_clean <= upper), np.nan)
    # Use forward-fill then backward-fill for better continuity
    filtered = filtered.ffill().bfill()
    return filtered

def calculate_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic percentage returns for price, liquidity, and volume.
    
    Computes period-over-period percentage changes for the core AMM pool metrics.
    Applies IQR-based outlier removal to maintain data quality while preserving
    important price movements and liquidity dynamics.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['price', 'liquidity', 'volumeUSD']
        
    Returns:
        pd.DataFrame: Input DataFrame with added columns:
        - 'price_return': Price percentage change (pct_change)
        - 'liquidity_return': Liquidity percentage change (pct_change) 
        - 'volume_return': Volume percentage change (pct_change)
        
    Note:
        All return series undergo IQR-based outlier removal with k=3.0 for robustness.
        Returns are essential base features for all subsequent technical indicators.
    """
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
    """
    Calculate volatility features with proper temporal alignment to prevent data leakage.
    
    Computes rolling standard deviation of returns over multiple time horizons (3h, 6h, 24h).
    Price volatility uses current data (no leakage), while liquidity and volume volatility
    are shifted by 1 period to avoid future information leakage when predicting these targets.
    
    Args:
        df (pd.DataFrame): DataFrame with return columns ['price_return', 'liquidity_return', 'volume_return']
        
    Returns:
        pd.DataFrame: Input DataFrame with added volatility columns:
        
        Price Volatility (no shift - price is exogenous):
        - 'price_volatility_3h/6h/24h': Rolling standard deviation of price returns
        
        Liquidity Volatility (shifted to avoid leakage):
        - 'liquidity_volatility_3h/6h/24h': Rolling std of liquidity returns (shifted by 1)
        
        Volume Volatility (shifted to avoid leakage):
        - 'volume_volatility_3h/6h/24h': Rolling std of volume returns (shifted by 1)
        
    Note:
        Volatility features are crucial for risk modeling and regime detection.
        Temporal shifts ensure model doesn't have access to future target information.
    """
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
    """
    Calculate moving average features with proper temporal alignment to prevent data leakage.
    
    Computes rolling mean of returns over multiple time horizons (3h, 6h, 24h).
    Moving averages help capture trend direction and momentum. Price moving averages
    use current data, while liquidity and volume averages are temporally shifted.
    
    Args:
        df (pd.DataFrame): DataFrame with return columns ['price_return', 'liquidity_return', 'volume_return']
        
    Returns:
        pd.DataFrame: Input DataFrame with added moving average columns:
        
        Price Moving Averages (no shift - price is exogenous):
        - 'price_ma_3h/6h/24h': Rolling mean of price returns
        
        Liquidity Moving Averages (shifted to avoid leakage):
        - 'liquidity_ma_3h/6h/24h': Rolling mean of liquidity returns (shifted by 1)
        
        Volume Moving Averages (shifted to avoid leakage):
        - 'volume_ma_3h/6h/24h': Rolling mean of volume returns (shifted by 1)
        
    Note:
        Moving averages smooth out short-term fluctuations and reveal underlying trends.
        Used as inputs for mean reversion and momentum indicators.
    """
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
    """
    Calculate cross-asset interaction features to capture market microstructure relationships.
    
    Computes correlations and ratios between different return series to identify
    co-movement patterns and efficiency relationships. These features capture
    how price, volume, and liquidity interact in AMM pools.
    
    Args:
        df (pd.DataFrame): DataFrame with return columns and moving averages
        
    Returns:
        pd.DataFrame: Input DataFrame with added cross-asset columns:
        
        Price-Volume/Liquidity Correlations:
        - 'price_volume_corr_6h/24h': Rolling correlation between price and volume returns
        - 'price_liquidity_corr_6h': Rolling correlation between price and liquidity returns
        
        Volume-Liquidity Efficiency Metrics:
        - 'volume_liquidity_return_ratio': Instantaneous volume/liquidity return ratio
        - 'volume_liquidity_ratio_ma_6h': 6-hour MA of volume/liquidity ratio
        - 'volume_liquidity_ratio_volatility': 6-hour volatility of volume/liquidity ratio
        
    Note:
        Cross-asset features help identify market stress, efficiency breakdowns,
        and regime changes. Liquidity and volume features are shifted to prevent leakage.
    """
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
    """
    Calculate market microstructure features to capture trading dynamics and market quality.
    
    Computes proxy measures for bid-ask spreads, market depth, and price impact
    using available AMM pool data. These features help identify market stress,
    liquidity provision efficiency, and trading cost dynamics.
    
    Args:
        df (pd.DataFrame): DataFrame with volatility features and return columns
        
    Returns:
        pd.DataFrame: Input DataFrame with added microstructure columns:
        
        Bid-Ask Spread Proxies:
        - 'price_volatility_to_volume_ratio': Price volatility scaled by volume activity
        - 'liquidity_return_depth_proxy': Liquidity depth relative to price volatility
        
        Market Impact Indicators:
        - 'market_impact_proxy': Volume impact on price movements (inverse relationship)
        - 'price_efficiency': Price volatility relative to volume volatility
        
    Note:
        Microstructure features help assess market quality and trading conditions.
        Higher ratios often indicate wider spreads or reduced market efficiency.
        Volume and liquidity features use shifted data to prevent target leakage.
    """
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
    """
    Calculate momentum and mean reversion features across multiple time horizons.
    
    Computes cumulative returns (momentum) and deviations from moving averages
    (mean reversion signals) for price, liquidity, and volume. These features
    capture persistent directional movements and reversal patterns.
    
    Args:
        df (pd.DataFrame): DataFrame with return columns and moving averages
        
    Returns:
        pd.DataFrame: Input DataFrame with added momentum columns:
        
        Price Momentum (no shift - price is exogenous):
        - 'price_momentum_3h/12h/24h': Cumulative price returns over different horizons
        - 'price_deviation_from_ma_6h/24h': Price return deviation from moving averages
        
        Liquidity Momentum (shifted to avoid leakage):
        - 'liquidity_momentum_6h/24h': Cumulative liquidity returns (shifted by 1)
        - 'liquidity_deviation_from_ma_6h': Liquidity deviation from moving average
        
        Volume Momentum (shifted to avoid leakage):
        - 'volume_momentum_6h/24h': Cumulative volume returns (shifted by 1)
        
    Note:
        Momentum features help identify trending behavior, while deviation features
        capture mean reversion opportunities. Essential for regime classification.
    """
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
    """
    Calculate volatility regime features to identify market stress and clustering patterns.
    
    Identifies high volatility periods and volatility clustering using percentile-based
    thresholds and persistence measures. These features help classify market regimes
    and predict volatility transitions.
    
    Args:
        df (pd.DataFrame): DataFrame with volatility features
        
    Returns:
        pd.DataFrame: Input DataFrame with added volatility regime columns:
        
        Volatility Regime Indicators:
        - 'high_vol_regime_3h/24h': Binary indicators for high volatility periods
          (above 80th percentile of weekly rolling window)
        - 'vol_cluster_indicator': Binary indicator for volatility clustering
          (consecutive high volatility periods)
        
        Volatility Term Structure:
        - 'vol_term_structure': Ratio of short-term to long-term volatility
          (3h volatility / 24h volatility)
        
    Note:
        Regime features use historical data (shifted by 1 period) to avoid leakage.
        High volatility regimes often persist and affect liquidity provision strategies.
        Volatility clustering indicates market stress or uncertainty periods.
    """
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
    """
    Calculate liquidity-specific features using stationary return-based measures.
    
    Computes liquidity concentration, percentile rankings, stress indicators, and
    stability measures. These features help assess liquidity supply dynamics and
    identify periods of liquidity stress or abundance that affect LP performance.
    
    Args:
        df (pd.DataFrame): DataFrame with liquidity return and volatility features
        
    Returns:
        pd.DataFrame: Input DataFrame with added liquidity columns:
        
        Liquidity Concentration & Ranking:
        - 'liquidity_return_concentration': Current liquidity relative to weekly maximum
        - 'liquidity_return_percentile_6h/24h': Percentile rank of liquidity returns
        
        Liquidity Stress Indicators:
        - 'liquidity_stress': Binary indicator for severe liquidity drops (bottom 10th percentile)
        - 'liquidity_abundance': Binary indicator for large liquidity increases (top 10th percentile)
        
        Liquidity Stability:
        - 'liquidity_stability': Inverse of liquidity volatility (stability measure)
        
    Note:
        All liquidity features use shifted data (lag 1) to prevent target leakage.
        Uses return-based measures for stationarity and comparability across pools.
        Liquidity stress periods often precede increased impermanent loss.
    """
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
    """
    Calculate time-based cyclical features and market session indicators.
    
    Extracts temporal patterns from datetime information using both linear and
    cyclical encodings. Captures market session effects, weekly patterns, and
    seasonal variations that influence AMM pool activity and trading patterns.
    
    Args:
        df (pd.DataFrame): DataFrame with 'datetime' column
        
    Returns:
        pd.DataFrame: Input DataFrame with added temporal columns:
        
        Basic Temporal Features:
        - 'hour': Hour of day (0-23)
        - 'day_of_week': Day of week (0=Monday, 6=Sunday)
        - 'month': Month of year (1-12)
        - 'season': Season encoding (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        
        Cyclical Encodings (preserves periodicity):
        - 'hour_sin/cos': Sine/cosine encoding of hour (24-hour cycle)
        - 'day_sin/cos': Sine/cosine encoding of day (7-day cycle)
        
        Market Session Indicators:
        - 'us_market_hours': Binary indicator for US trading hours (9 AM - 4 PM EST)
        - 'asia_market_hours': Binary indicator for Asia trading hours
        - 'weekend': Binary indicator for weekend periods
        
    Note:
        Cyclical encodings ensure that temporal boundaries (e.g., 23:59 to 00:00)
        are treated as adjacent. Market session features capture time-zone-specific
        trading patterns that affect liquidity and volume dynamics.
    """
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
    """
    Calculate advanced technical indicators including RSI and Bollinger Band-like features.
    
    Computes momentum oscillators and volatility bands commonly used in technical
    analysis, adapted for AMM pool data. These indicators help identify overbought/
    oversold conditions and volatility expansion/contraction patterns.
    
    Args:
        df (pd.DataFrame): DataFrame with return columns, moving averages, and volatility features
        
    Returns:
        pd.DataFrame: Input DataFrame with added technical indicator columns:
        
        RSI (Relative Strength Index) Indicators:
        - 'price_rsi_6h/24h': Price RSI over 6-hour and 24-hour windows
        - 'liquidity_rsi_6h': Liquidity RSI over 6-hour window (shifted to avoid leakage)
        - 'volume_rsi_6h': Volume RSI over 6-hour window (shifted to avoid leakage)
        
        Bollinger Band-like Features:
        - 'price_bb_upper_6h/lower_6h': Upper/lower volatility bands (MA ± 2*std)
        - 'price_bb_position': Normalized position within volatility bands (0-1 scale)
        
    Note:
        RSI values range from 0-100, with 70+ indicating overbought and 30- oversold.
        Bollinger band position shows relative price level within recent volatility range.
        Liquidity and volume indicators use shifted data to prevent target leakage.
    """
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
    """
    Calculate shock and event detection features to identify extreme market conditions.
    
    Identifies sudden price movements, volume spikes, and liquidity events that
    significantly deviate from normal market behavior. These features help detect
    market stress, news events, and other anomalous conditions affecting LP returns.
    
    Args:
        df (pd.DataFrame): DataFrame with return, volatility, and percentile features
        
    Returns:
        pd.DataFrame: Input DataFrame with added shock detection columns:
        
        Price Shock Indicators:
        - 'price_shock_3h/6h': Binary indicators for extreme price movements
          (absolute return > 3 standard deviations from rolling volatility)
        
        Volume Event Indicators:
        - 'volume_spike': Binary indicator for extreme volume increases 
          (above 95th percentile of 24-hour rolling window)
        - 'volume_drought': Binary indicator for extreme volume decreases
          (below 5th percentile of 24-hour rolling window)
        
        Liquidity Event Indicators:
        - 'liquidity_drain': Binary indicator for severe liquidity decreases (>10% drop)
        - 'liquidity_injection': Binary indicator for large liquidity increases (>10% gain)
        
    Note:
        Shock detection uses statistical thresholds based on rolling windows.
        Volume and liquidity events use shifted data to prevent target leakage.
        These features are crucial for risk management and regime classification.
    """
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

