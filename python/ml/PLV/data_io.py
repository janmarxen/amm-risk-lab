"""
plv_data_io.py

Utility functions and PyTorch Dataset for loading, engineering, and preparing Uniswap V3 pool data for ML models.
"""
import pandas as pd
import time
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from python.utils.subgraph_utils import fetch_pool_hourly_data

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a pool DataFrame, including returns, rolling stats, and temporal features.
    """
    df = df.copy()
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['periodStartUnix'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)
    # Returns
    df['price_return'] = df['price'].pct_change()
    df['liquidity_return'] = df['liquidity'].pct_change()
    df['volume_return'] = df['volumeUSD'].pct_change()
    # Outlier removal
    def remove_outliers_iqr(series, k=3.0):
        q_01 = series.quantile(0.3)
        q_90 = series.quantile(0.6)
        iqr = q_90 - q_01
        lower = q_01 - k * iqr
        upper = q_90 + k * iqr
        return series.where((series >= lower) & (series <= upper))
    for col in ['price_return', 'liquidity_return', 'volume_return']:
        if col in df.columns:
            df[col] = remove_outliers_iqr(df[col])
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    # Price-based volatility and moving averages
    df['price_volatility_3h'] = df['price_return'].rolling(window=3).std()
    df['price_volatility_6h'] = df['price_return'].rolling(window=6).std()
    df['price_volatility_24h'] = df['price_return'].rolling(window=24).std()
    df['price_ma_3h'] = df['price_return'].rolling(window=3).mean()
    df['price_ma_6h'] = df['price_return'].rolling(window=6).mean()
    df['price_ma_24h'] = df['price_return'].rolling(window=24).mean()
    # Liquidity-based volatility and moving averages (shifted by 1 to avoid lookahead bias)
    df['liquidity_volatility_3h'] = df['liquidity_return'].shift(1).rolling(window=3).std()
    df['liquidity_volatility_6h'] = df['liquidity_return'].shift(1).rolling(window=6).std()
    df['liquidity_volatility_24h'] = df['liquidity_return'].shift(1).rolling(window=24).std()
    df['liquidity_ma_3h'] = df['liquidity_return'].shift(1).rolling(window=3).mean()
    df['liquidity_ma_6h'] = df['liquidity_return'].shift(1).rolling(window=6).mean()
    df['liquidity_ma_24h'] = df['liquidity_return'].shift(1).rolling(window=24).mean()
    # Volume-based volatility and moving averages (shifted by 1 to avoid lookahead bias)
    df['volume_volatility_3h'] = df['volume_return'].shift(1).rolling(window=3).std()
    df['volume_volatility_6h'] = df['volume_return'].shift(1).rolling(window=6).std()
    df['volume_volatility_24h'] = df['volume_return'].shift(1).rolling(window=24).std()
    df['volume_ma_3h'] = df['volume_return'].shift(1).rolling(window=3).mean()
    df['volume_ma_6h'] = df['volume_return'].shift(1).rolling(window=6).mean()
    df['volume_ma_24h'] = df['volume_return'].shift(1).rolling(window=24).mean()
    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    df['season'] = df['month'].apply(get_season)
    if 'periodStartUnix' in df.columns:
        df = df.drop(columns=['periodStartUnix'])
    return df

def add_lagged_features(df: pd.DataFrame, n_lags: int = 3, lag_features: List[str] = None) -> pd.DataFrame:
    """
    Add lagged versions of selected features to the DataFrame.
    """
    df = df.copy()
    if lag_features is None:
        lag_features = ['price_return', 'liquidity_return']
    for feature in lag_features:
        if feature in df.columns:
            for lag in range(1, n_lags + 1):
                df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    return df

def dropna_lstm(df: pd.DataFrame, features: List[str], target_col: str) -> pd.DataFrame:
    """
    Drop rows with NaNs in any of the selected features or target column.
    """
    mask = df[features + [target_col]].notnull().all(axis=1)
    return df.loc[mask].reset_index(drop=True)

def get_X_y(df: pd.DataFrame, features: List[str], target_col: str, n_lags: int):
    """
    Convert a DataFrame to supervised learning arrays for LSTM: (X, y_cls, y_reg).
    X: lagged feature windows, y_cls: zero-class labels, y_reg: regression targets.
    """
    X, y_cls, y_reg = [], [], []
    for i in range(n_lags, len(df)):
        X.append(df[features].iloc[i-n_lags:i].values)
        target = df[target_col].iloc[i]
        y_cls.append(1 if target == 0 else 0)
        y_reg.append(float(target))
    return X, y_cls, y_reg

def fetch_and_save_pools(
    api_key: str,
    subgraph_id: str,
    pool_addresses: List[str],
    start_date: str,
    end_date: str,
    hdf5_path: str,
    min_rows: int = 100,
    mode: str = 'w'  # 'w' = overwrite pool, 'a' = append/update pool, 'x' = skip if pool exists
):
    """
    Fetch hourly data for each pool, apply feature engineering, and save to HDF5.
    Each pool is saved under key /pool_<address>. Metadata is saved under /meta.
    mode: 'w' (overwrite pool), 'a' (append/update pool), 'x' (skip if pool exists)
    """
    # Always open the HDF5 file in append mode
    store = pd.HDFStore(hdf5_path, mode='a')
    fetched = []
    total = len(pool_addresses)
    for idx, addr in enumerate(pool_addresses, 1):
        pool_key = f'pool_{addr.lower()}'
        if mode == 'x' and pool_key in store:
            print(f"[{idx}/{total}] Skipping {addr}: already exists in {hdf5_path}")
            continue
        df = fetch_pool_hourly_data(api_key, subgraph_id, addr, start_date, end_date)
        n = len(df)
        if df is not None and n >= min_rows:
            print(f"[{idx}/{total}] Fetching {addr} with {n} rows")
            df = feature_engineer(df)
            store.put(pool_key, df, format='table')
            fetched.append(addr.lower())
        else:
            print(f"[{idx}/{total}] Skipping {addr}: insufficient data")
    # Save metadata
    meta = pd.DataFrame({
        'pool_addresses': [fetched],
        'fetch_time': [time.time()]
    })
    store.put('meta', meta)
    store.close()
    print(f"Saved {len(fetched)} pools to {hdf5_path}")

def load_pool_data(hdf5_path: str, pool_address: str) -> pd.DataFrame:
    """
    Load a single pool's data from HDF5.
    """
    with pd.HDFStore(hdf5_path, mode='r') as store:
        key = f'pool_{pool_address.lower()}'
        return store[key]

def load_all_pools(hdf5_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all pools' data from HDF5 as a dict of address -> DataFrame.
    """
    with pd.HDFStore(hdf5_path, mode='r') as store:
        pools = [k[1:] for k in store.keys() if k.startswith('/pool_')]
        return {p[5:]: store[p] for p in pools}

def get_saved_pool_addresses(hdf5_path: str) -> List[str]:
    """
    Return the list of pool addresses saved in the HDF5 file.
    """
    with pd.HDFStore(hdf5_path, mode='r') as store:
        if 'meta' in store:
            meta = store['meta']
            return list(meta['pool_addresses'].iloc[0])
        pools = [k[1:] for k in store.keys() if k.startswith('/pool_')]
        return [p[5:] for p in pools]

class LPsDataset(Dataset):
    """
    PyTorch Dataset for LSTM pretraining/fine-tuning on multiple pools from HDF5.
    Each item is a tuple (X, y_cls, y_reg) for supervised learning.
    Allows flexible split: 'train', 'val', or 'test' using split_dates dict.
    """
    def __init__(
        self,
        hdf5_path: str,
        pool_addresses: list = None,
        features: list = None,
        target: str = None,
        n_lags: int = 1,
        split: str = 'train',
        split_dates: dict = None,
        lag_features=None,
        verbose: int = 1
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file.
            pool_addresses: List of pool addresses to load. If None, loads all saved pool addresses.
            features: List of feature columns to use.
            target: Name of target column.
            n_lags: Number of lag steps for LSTM.
            split: 'train', 'val', or 'test'.
            split_dates: Dict with keys 'train_end', 'val_end' for splitting.
            lag_features: List of features to lag (default: price_return, liquidity_return).
            verbose: Print progress if 1.
        """
        if pool_addresses is None:
            pool_addresses = get_saved_pool_addresses(hdf5_path)
        self.X, self.y_cls, self.y_reg = [], [], []
        total = len(pool_addresses)
        for idx, addr in enumerate(pool_addresses, 1):
            try:
                df = load_pool_data(hdf5_path, addr)
            except (KeyError, FileNotFoundError):
                if verbose:
                    print(f"{idx}/{total}: Pool {addr} not found in HDF5, skipping.")
                continue
            except Exception as e:
                if verbose:
                    print(f"{idx}/{total}: Error loading pool {addr}: {e}")
                continue
            # Flexible split logic
            if split_dates is not None:
                if split == 'train':
                    end = split_dates.get('train_end', None)
                    if end is not None:
                        df = df[df['datetime'] <= pd.to_datetime(end)].copy()
                elif split == 'val':
                    start = split_dates.get('train_end', None)
                    end = split_dates.get('val_end', None)
                    if start is not None:
                        df = df[df['datetime'] > pd.to_datetime(start)]
                    if end is not None:
                        df = df[df['datetime'] <= pd.to_datetime(end)]
                    df = df.copy()
                elif split == 'test':
                    start = split_dates.get('val_end', None)
                    if start is not None:
                        df = df[df['datetime'] > pd.to_datetime(start)].copy()
            df = add_lagged_features(df, n_lags=n_lags, lag_features=lag_features)
            df['pool'] = addr
            df = dropna_lstm(df, features, target)
            X, y_cls, y_reg = get_X_y(df, features, target, n_lags)
            self.X.extend(X)
            self.y_cls.extend(y_cls)
            self.y_reg.extend(y_reg)
            if verbose:
                print(f"{idx}/{total}: Pool Dataset processed")
        if self.X:
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y_cls = torch.tensor(self.y_cls, dtype=torch.float32)
            self.y_reg = torch.tensor(self.y_reg, dtype=torch.float32)
        else:
            self.X = torch.empty((0, n_lags, len(features)), dtype=torch.float32)
            self.y_cls = torch.empty((0,), dtype=torch.float32)
            self.y_reg = torch.empty((0,), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """Return (X, y_cls, y_reg) tuple for sample idx."""
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]
