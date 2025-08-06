"""
data_io.py

Utility functions and PyTorch Dataset for loading, engineering, and preparing Uniswap V3 pool data for ML models.
"""
import pandas as pd
import time
from typing import List, Dict, Tuple
import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler
import random

import torch
from torch.utils.data import Dataset

import concurrent.futures
from threading import Lock

from python.utils.subgraph_utils import fetch_pool_hourly_data, fetch_pools_hourly_data_batched, fetch_pools_hourly_data_batched_parallel

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a pool DataFrame, including returns, rolling stats, and temporal features.
    Args:
        df (pd.DataFrame): Raw pool data.
    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    df = df.copy()
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['periodStartUnix'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)
    df['price'] = df['price'].astype(np.float64)
    df['liquidity'] = df['liquidity'].astype(np.float64)
    df['volumeUSD'] = df['volumeUSD'].astype(np.float64)
    # Returns
    df['price_return'] = df['price'].pct_change().astype(np.float64)
    df['liquidity_return'] = df['liquidity'].pct_change().astype(np.float64)
    df['volume_return'] = df['volumeUSD'].pct_change().astype(np.float64)
    # Outlier removal with improved precision
    def remove_outliers_iqr(series, k=3.0):
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
    # Apply to desired columns
    for col in ['price_return', 'liquidity_return', 'volume_return']:
        if col in df.columns:
            df[col] = remove_outliers_iqr(df[col])

    ### Should shift be 1 or -1? ### 
    # Price-based volatility and moving averages (shifted by 1 to avoid lookahead bias)
    df['price_volatility_3h'] = df['price_return'].shift(1).rolling(window=3).std()
    df['price_volatility_6h'] = df['price_return'].shift(1).rolling(window=6).std()
    df['price_volatility_24h'] = df['price_return'].shift(1).rolling(window=24).std()
    df['price_ma_3h'] = df['price_return'].shift(1).rolling(window=3).mean()
    df['price_ma_6h'] = df['price_return'].shift(1).rolling(window=6).mean()
    df['price_ma_24h'] = df['price_return'].shift(1).rolling(window=24).mean()
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

def dropna(df: pd.DataFrame, features: List[str], target_col: str) -> pd.DataFrame:
    """
    Drop rows with NaNs in any of the selected features or target column.
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (List[str]): List of feature columns.
        target_col (str): Target column name.
    Returns:
        pd.DataFrame: DataFrame with rows containing NaNs dropped.
    """
    mask = df[features + [target_col]].notnull().all(axis=1)
    return df.loc[mask].reset_index(drop=True)

# def get_X_y(df: pd.DataFrame, features: List[str], target_col: str, n_lags: int) -> Tuple[list, list, list]:
#     """
#     Convert a DataFrame to supervised learning arrays for ZeroInflated LSTM/Transformer:
#     - X: lagged feature windows + target lags
#     - y_cls: classification label (1 if target == 0, else 0)
#     - y_reg: regression target

#     Faster by leveraging NumPy vectorization.
#     """
#     # Drop rows with missing/infinite values first
#     df = df.copy()
#     df = df.replace([np.inf, -np.inf], np.nan)
#     df = df.dropna(subset=features + [target_col])

#     data_feats = df[features].to_numpy()
#     data_target = df[target_col].to_numpy()

#     T = len(df)
    
#     if T < n_lags + 1:  # Need at least n_lags+1 points for this alignment
#         return [], [], []
    
#     # Use sliding window for features, INCLUDING present timestep
#     feats_window = np.lib.stride_tricks.sliding_window_view(data_feats, (n_lags, data_feats.shape[1]))
#     feats_window = feats_window[:, 0, :, :]  # shape: (T - n_lags + 1, n_lags, n_features)

#     # Use sliding window for target lags, EXCLUDING present timestep (only past values)
#     target_lags_past = np.lib.stride_tricks.sliding_window_view(data_target, n_lags-1)
#     # shape: (T - n_lags + 2, n_lags-1)
    
#     # Get FUTURE target values (y) - predict t+1 using features up to t
#     y = data_target[n_lags:]  # Predict target at t+1, using features from [t-n_lags+1, ..., t]
    
#     # Pad target lags to match n_lags sequence length expected by model
#     target_lags_expanded = target_lags_past.reshape(target_lags_past.shape[0], n_lags-1, 1)
#     # Pad with zeros for the "missing" present timestep to maintain model input shape
#     zero_pad = np.zeros((target_lags_expanded.shape[0], 1, 1))
#     target_lags_padded = np.concatenate([target_lags_expanded, zero_pad], axis=1)  # shape: (N, n_lags, 1)

#     # Align arrays: target_lags_past starts 1 step later than feats_window
#     # feats_window: indices [0, 1, 2, ..., T-n_lags] -> features for times [0:n_lags, 1:n_lags+1, ..., T-n_lags:T]
#     # target_lags_past: indices [0, 1, 2, ..., T-n_lags+1] -> target lags for times [0:n_lags-1, 1:n_lags, ..., T-n_lags+1:T-1]
#     # y: indices [0, 1, 2, ..., T-n_lags] -> targets for times [n_lags, n_lags+1, ..., T-1]
    
#     # Cut arrays to same final length
#     min_len = min(len(feats_window), len(target_lags_padded), len(y))
#     feats_window = feats_window[:min_len]  # Take first min_len samples
#     target_lags_padded = target_lags_padded[1:min_len+1]  # Shift by 1 to align properly  
#     y = y[:min_len]

#     # Concatenate features and target lags  
#     X = np.concatenate([feats_window, target_lags_padded], axis=2)  # shape: (N, n_lags, f+1)

#     # Filter finite rows
#     mask = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
#     X = X[mask]
#     y_reg = y[mask]
#     y_cls = (y_reg == 0).astype(float)

#     return X.tolist(), y_cls.tolist(), y_reg.tolist()

def get_X_y(df: pd.DataFrame, features: List[str], target_col: str, n_lags: int) -> Tuple[list, list, list]:
    """
    Convert a DataFrame to supervised learning arrays for ZeroInflated LSTM/Transformer:
    - X: lagged features (ending at t) + target lags (ending at t-1)
    - y_cls: classification label (1 if target[t] == 0, else 0)
    - y_reg: regression target (at time t)
    
    Features at time t are valid - they don't leak information about target[t].
    """

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + [target_col])

    data_feats = df[features].to_numpy()
    data_target = df[target_col].to_numpy()
    T = len(df)

    if T < n_lags + 1:
        return [], [], []

    # Feature window: t - n_lags + 1 to t (length = n_lags, INCLUDE present)
    feats_window = np.lib.stride_tricks.sliding_window_view(data_feats, (n_lags, data_feats.shape[1]))
    feats_window = feats_window[:, 0, :, :]  # shape: (T - n_lags + 1, n_lags, num_features)

    # Target lag window: t - n_lags to t - 1 (length = n_lags, EXCLUDE present)
    target_lags = np.lib.stride_tricks.sliding_window_view(data_target, n_lags + 1)
    target_lags = target_lags[:, :-1]  # Remove value at t
    target_lags = target_lags[:, :, np.newaxis]  # shape: (N, n_lags, 1)

    # Target at time t
    y = data_target[n_lags:]

    # Match lengths
    min_len = min(len(feats_window), len(target_lags), len(y))
    feats_window = feats_window[-min_len:]
    target_lags = target_lags[-min_len:]
    y = y[-min_len:]

    # Concatenate features and target lags
    X = np.concatenate([feats_window, target_lags], axis=2)  # shape: (N, n_lags, num_features + 1)

    # Filter valid
    mask = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
    X = X[mask]
    y_reg = y[mask]
    y_cls = (y_reg == 0).astype(float)

    return X.tolist(), y_cls.tolist(), y_reg.tolist()


def fetch_and_save_pools(
    api_key: str,
    subgraph_id: str,
    pool_addresses: List[str],
    start_date: str,
    end_date: str,
    hdf5_path: str,
    min_rows: int = 100,
    mode: str = 'w',  # 'w' = overwrite pool, 'a' = append/update pool, 'x' = skip if exists
    fetch_mode: str = 'parallel',  # 'sequential', 'batched', or 'parallel'
    max_workers: int = 16  # Only used for parallel mode
):
    """
    Fetch hourly data for each pool, apply feature engineering, and save to HDF5.
    Each pool is saved under key /pool_<address>. Metadata is saved under /meta.
    Args:
        api_key (str): The Graph API key.
        subgraph_id (str): Subgraph ID.
        pool_addresses (List[str]): List of pool addresses.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        hdf5_path (str): Path to HDF5 file.
        min_rows (int): Minimum number of rows required to save pool.
        mode (str): 'w' (overwrite), 'a' (append/update), 'x' (skip if exists).
        fetch_mode (str): 'sequential', 'batched', or 'parallel' (default: 'parallel').
        max_workers (int): Number of threads for parallel mode.
    Returns:
        None
    """
    fetched = []
    total = len(pool_addresses)
    # Open HDF5 file in append mode
    with h5py.File(hdf5_path, 'a') as h5f:
        # Fetch all pools according to fetch_mode
        if fetch_mode == 'sequential':
            pool_data_dict = {}
            for addr in pool_addresses:
                df = fetch_pool_hourly_data(api_key, subgraph_id, addr, start_date, end_date)
                pool_data_dict[addr] = df
        elif fetch_mode == 'batched':
            pool_data_dict = fetch_pools_hourly_data_batched(api_key, subgraph_id, pool_addresses, start_date, end_date)
        else:  # 'parallel' (default)
            pool_data_dict = fetch_pools_hourly_data_batched_parallel(api_key, subgraph_id, pool_addresses, start_date, end_date, max_workers=max_workers)
        for idx, addr in enumerate(pool_addresses, 1):
            pool_key = f'pool_{addr.lower()}'
            if mode == 'x' and pool_key in h5f:
                print(f"[{idx}/{total}] Skipping {addr}: already exists in {hdf5_path}")
                continue
            df = pool_data_dict.get(addr, pd.DataFrame())
            n = len(df)
            if df is not None and n >= min_rows:
                print(f"[{idx}/{total}] Fetching {addr} with {n} rows")
                df = feature_engineer(df)
                # Split columns by dtype
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                str_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                grp = h5f.require_group(pool_key)
                # Remove existing datasets if overwriting
                if pool_key in h5f and mode == 'w':
                    for k in list(grp.keys()):
                        del grp[k]
                # Save numeric data
                if num_cols:
                    grp.create_dataset('data', data=df[num_cols].to_numpy(), compression='gzip', chunks=True)
                    dt = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset('num_columns', data=np.array(num_cols, dtype=object), dtype=dt)
                # Save string/object data
                if str_cols:
                    str_data = df[str_cols].astype(str).to_numpy()
                    dt = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset('strings', data=str_data, dtype=dt, compression='gzip', chunks=True)
                    grp.create_dataset('str_columns', data=np.array(str_cols, dtype=object), dtype=dt)
                fetched.append(addr.lower())
            else:
                print(f"[{idx}/{total}] Skipping {addr}: insufficient data")
            # Save metadata
            meta_grp = h5f.require_group('meta')
            meta_grp.attrs['pool_addresses'] = ','.join(fetched)
            meta_grp.attrs['fetch_time'] = time.time()
    print(f"Saved {len(fetched)} pools to {hdf5_path}")

def load_pool_data(hdf5_path: str, pool_address: str) -> pd.DataFrame:
    """
    Load a single pool's data from HDF5 using h5py.
    Args:
        hdf5_path (str): Path to HDF5 file.
        pool_address (str): Pool address.
    Returns:
        pd.DataFrame: DataFrame for the pool.
    Raises:
        KeyError: If pool not found.
        ValueError: If no data found for pool.
    """
    pool_key = f'pool_{pool_address.lower()}'
    with h5py.File(hdf5_path, 'r') as h5f:
        if pool_key not in h5f:
            raise KeyError(f"Pool {pool_address} not found in {hdf5_path}")
        grp = h5f[pool_key]
        dfs = []
        # Numeric columns
        if 'data' in grp and 'num_columns' in grp:
            data = grp['data'][()]
            num_columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in grp['num_columns'][()]]
            dfs.append(pd.DataFrame(data, columns=num_columns))
        # String columns
        if 'strings' in grp and 'str_columns' in grp:
            str_data = grp['strings'][()]
            str_columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in grp['str_columns'][()]]
            dfs.append(pd.DataFrame(str_data, columns=str_columns))
        if dfs:
            df = pd.concat(dfs, axis=1)
            # Convert 'datetime' column to pandas datetime if present, via string
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'].astype(str), errors='coerce')
            return df
        else:
            raise ValueError(f"No data found for pool {pool_address} in {hdf5_path}")

def load_selected_pools_in_memory(hdf5_path: str, pool_addresses: list) -> Dict[str, pd.DataFrame]:
    """
    Load only the specified pools' data from HDF5 as a dict of address -> DataFrame, kept in memory.
    Args:
        hdf5_path (str): Path to HDF5 file.
        pool_addresses (list): List of pool addresses to load.
    Returns:
        Dict[str, pd.DataFrame]: Mapping pool address to DataFrame.
    """
    pool_dict = {}
    with h5py.File(hdf5_path, 'r') as h5f:
        for addr in pool_addresses:
            key = f'pool_{addr.lower()}'
            if key in h5f:
                grp = h5f[key]
                dfs = []
                if 'data' in grp and 'num_columns' in grp:
                    data = grp['data'][()]
                    num_columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in grp['num_columns'][()]]
                    dfs.append(pd.DataFrame(data, columns=num_columns))
                if 'strings' in grp and 'str_columns' in grp:
                    str_data = grp['strings'][()]
                    str_columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in grp['str_columns'][()]]
                    dfs.append(pd.DataFrame(str_data, columns=str_columns))
                if dfs:
                    df = pd.concat(dfs, axis=1)
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'].astype(str), errors='coerce')
                    pool_dict[addr] = df
    return pool_dict

def load_all_pools_in_memory(hdf5_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all pools' data from HDF5 as a dict of address -> DataFrame, kept in memory, using h5py.
    Args:
        hdf5_path (str): Path to HDF5 file.
    Returns:
        Dict[str, pd.DataFrame]: Mapping pool address to DataFrame.
    """
    pool_dict = {}
    with h5py.File(hdf5_path, 'r') as h5f:
        for key in h5f.keys():
            if key.startswith('pool_'):
                addr = key[5:]
                grp = h5f[key]
                dfs = []
                if 'data' in grp and 'num_columns' in grp:
                    data = grp['data'][()]
                    num_columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in grp['num_columns'][()]]
                    dfs.append(pd.DataFrame(data, columns=num_columns))
                if 'strings' in grp and 'str_columns' in grp:
                    str_data = grp['strings'][()]
                    str_columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in grp['str_columns'][()]]
                    dfs.append(pd.DataFrame(str_data, columns=str_columns))
                if dfs:
                    df = pd.concat(dfs, axis=1)
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'].astype(str), errors='coerce')
                    pool_dict[addr] = df
    return pool_dict

def make_lps_dataset_from_pool_dict(
    pool_dict: Dict[str, pd.DataFrame],
    pool_addresses: list,
    features: list,
    target: str,
    n_lags: int,
    split: str,
    split_dates: dict,
    verbose: int = 1
) -> torch.utils.data.Dataset:
    """
    Construct LPsDataset from a dict of DataFrames, avoiding disk reads.
    Args:
        pool_dict (Dict[str, pd.DataFrame]): Mapping pool address to DataFrame.
        pool_addresses (list): List of pool addresses to include.
        features (list): List of feature columns.
        target (str): Target column name.
        n_lags (int): Number of lag steps.
        split (str): 'train', 'val', or 'test'.
        split_dates (dict): Dict with split start/end dates.
        verbose (int): Print progress if 1.
    Returns:
        torch.utils.data.Dataset: In-memory dataset.
    """
    class InMemoryLPsDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.X, self.y_cls, self.y_reg = [], [], []
            total = len(pool_addresses)
            for idx, addr in enumerate(pool_addresses, 1):
                if addr not in pool_dict:
                    if verbose:
                        print(f"{idx}/{total}: Pool {addr} not found in memory, skipping.")
                    continue
                df = pool_dict[addr]
                # Flexible split logic with custom start/end for each split
                if split_dates is not None:
                    if split == 'train':
                        start = split_dates.get('train_start', None)
                        end = split_dates.get('train_end', None)
                        if start is not None:
                            df = df[df['datetime'] >= pd.to_datetime(start)]
                        if end is not None:
                            df = df[df['datetime'] <= pd.to_datetime(end)]
                        df = df.copy()
                    elif split == 'val':
                        start = split_dates.get('val_start', None)
                        end = split_dates.get('val_end', None)
                        if start is not None:
                            df = df[df['datetime'] >= pd.to_datetime(start)]
                        if end is not None:
                            df = df[df['datetime'] <= pd.to_datetime(end)]
                        df = df.copy()
                    elif split == 'test':
                        start = split_dates.get('test_start', None)
                        end = split_dates.get('test_end', None)
                        if start is not None:
                            df = df[df['datetime'] >= pd.to_datetime(start)]
                        if end is not None:
                            df = df[df['datetime'] <= pd.to_datetime(end)]
                        df = df.copy()
                df['pool'] = addr
                df = dropna(df, features, target)
                X, y_cls, y_reg = get_X_y(df, features, target, n_lags)
                self.X.extend(X)
                self.y_cls.extend(y_cls)
                self.y_reg.extend(y_reg)
                if verbose:
                    print(f"{idx}/{total}: Pool {addr} Dataset processed")
            if self.X:
                self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
                self.y_cls = torch.tensor(self.y_cls, dtype=torch.float32)
                self.y_reg = torch.tensor(self.y_reg, dtype=torch.float32)
            else:
                self.X = torch.empty((0, n_lags, len(features) + 1), dtype=torch.float32)  # +1 for target lag dimension
                self.y_cls = torch.empty((0,), dtype=torch.float32)
                self.y_reg = torch.empty((0,), dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y_cls[idx], self.y_reg[idx]
    return InMemoryLPsDataset()

def get_saved_pool_addresses(hdf5_path: str) -> List[str]:
    """
    Return the list of pool addresses saved in the HDF5 file using h5py.
    Args:
        hdf5_path (str): Path to HDF5 file.
    Returns:
        List[str]: List of pool addresses.
    """
    with h5py.File(hdf5_path, 'r') as h5f:
        if 'meta' in h5f:
            meta_grp = h5f['meta']
            pool_addresses_str = meta_grp.attrs.get('pool_addresses', '')
            if isinstance(pool_addresses_str, bytes):
                pool_addresses_str = pool_addresses_str.decode('utf-8')
            return [addr for addr in pool_addresses_str.split(',') if addr]
        # Fallback: find all pool groups
        pools = [k for k in h5f.keys() if k.startswith('pool_')]
        return [p[5:] for p in pools]


def fit_scalers(
    hdf5_path: str,
    pool_addresses: list,
    features: list,
    target: str,
    split_dates: dict = None,
    verbose: int = 1,
    num_workers: int = 16,
    sample_size_pct: float = 0.1,
    feature_scaler=None,
    target_reg_scaler=None,
):
    """
    Fit feature and target scalers on a random sample of the data.
    Args:
        hdf5_path (str): Path to HDF5 file.
        pool_addresses (list): List of pool addresses to sample from.
        features (list): List of feature columns.
        target (str): Target column name.
        split_dates (dict): Dict with split start/end dates.
        verbose (int): Print progress if 1.
        num_workers (int): Number of threads for parallel loading.
        sample_size_pct (float): Fraction of pool_addresses to use (0 < pct <= 1).
    Returns:
        (feature_scaler, target_reg_scaler): Fitted StandardScaler objects.
    """
    if not (0 < sample_size_pct <= 1):
        raise ValueError("sample_size_pct must be in (0, 1]")
    n_sample = max(1, int(len(pool_addresses) * sample_size_pct))
    sample_addresses = random.sample(pool_addresses, n_sample)
    if verbose:
        print(f"[fit_scalers] Fitting scalers on {n_sample} pools ({sample_size_pct*100:.1f}% of total)")
    data_by_pool = load_selected_pools_in_memory(hdf5_path, sample_addresses)
    # Allow passing in existing scalers, else create new ones
    if feature_scaler is None:
        feature_scaler = StandardScaler()
    if target_reg_scaler is None:
        target_reg_scaler = StandardScaler()
    scaler_lock = Lock()

    def process_and_partial_fit(addr):
        df = data_by_pool.get(addr)
        if df is None or df.empty:
            return 0, 0
        # Time filtering (use train split if available)
        if split_dates:
            start, end = split_dates.get('train_start'), split_dates.get('train_end')
            if start:
                df = df[df['datetime'] >= pd.to_datetime(start)]
            if end:
                df = df[df['datetime'] <= pd.to_datetime(end)]
        df = df.dropna(subset=features + [target])
        if df.empty:
            return 0, 0
        X = df[features].values
        y = df[target].values.reshape(-1, 1)
        # Filter out non-finite rows
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).flatten()
        X = X[mask]
        y = y[mask]
        if X.shape[0] == 0 or y.shape[0] == 0:
            return 0, 0
        with scaler_lock:
            feature_scaler.partial_fit(X)
            target_reg_scaler.partial_fit(y)
        return X.shape[0], y.shape[0]

    total_X, total_y = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_and_partial_fit, addr) for addr in sample_addresses]
        for i, f in enumerate(futures, 1):
            n_x, n_y = f.result()
            total_X += n_x
            total_y += n_y
            if verbose:
                print(f"[fit_scalers] Processed {i}/{n_sample}: {n_x} feature, {n_y} target samples")
    if total_X == 0 or total_y == 0:
        raise ValueError("No data found for fitting scalers.")
    if verbose:
        print(f"[fit_scalers] Fitted feature scaler on {total_X} samples, target scaler on {total_y} samples")
    return feature_scaler, target_reg_scaler


class LPsDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        pool_addresses: List[str] = None,
        features: List[str] = None,
        target: str = None,
        n_lags: int = 1,
        split: str = 'train',
        split_dates: dict = None,
        feature_scaler=None,
        target_reg_scaler=None,
        verbose: int = 0,
        num_workers: int = 16
    ):
        self.X, self.y_cls, self.y_reg = [], [], []
        self.split = split
        self.feature_scaler = feature_scaler
        self.target_reg_scaler = target_reg_scaler
        # Load the pools' data into memory
        data_by_pool = load_selected_pools_in_memory(hdf5_path, pool_addresses)
        if pool_addresses is None:
            pool_addresses = list(data_by_pool.keys())
        total = len(pool_addresses)
        # Get start and end dates for the split
        if split_dates:
            if split == 'train':
                start, end = split_dates.get('train_start'), split_dates.get('train_end')
            elif split == 'val':
                start, end = split_dates.get('val_start'), split_dates.get('val_end')
            elif split == 'test':
                start, end = split_dates.get('test_start'), split_dates.get('test_end')
            else:
                start = end = None

        def filter_and_process(addr):
            df = data_by_pool.get(addr)
            if df is None or df.empty:
                return [], [], []
            # Time filtering
            if start:
                df = df[df['datetime'] >= pd.to_datetime(start)]
            if end:
                df = df[df['datetime'] <= pd.to_datetime(end)]
            # Drop rows with NaNs in features or target
            df = df.dropna(subset=features + [target])
            if df.empty:
                return [], [], []
            # Scale features and target independently per pool
            X_feats = df[features].values
            y_target = df[target].values.reshape(-1, 1)
            # Ensure no non-finite values in target
            if not np.all(np.isfinite(y_target)):
                if verbose:
                    print(f"[{addr}] Skipping due to bad target values: {y_target[~np.isfinite(y_target)]}")
                return [], [], []
            # Ensure no non-finite values in features
            if not np.all(np.isfinite(X_feats)):
                if verbose:
                    print(f"[{addr}] Skipping due to bad feature values: {X_feats[~np.isfinite(X_feats)]}")
                return [], [], []

            # Use passed-in scalers if provided, else fit per-pool
            if self.feature_scaler is not None:
                X_feats_scaled = self.feature_scaler.transform(X_feats)
            else:
                feature_scaler = StandardScaler()
                X_feats_scaled = feature_scaler.fit_transform(X_feats)

            if self.target_reg_scaler is not None:
                y_target_scaled = self.target_reg_scaler.transform(y_target).flatten()
            else:
                target_scaler = StandardScaler()
                y_target_scaled = target_scaler.fit_transform(y_target).flatten()

            df.loc[:, features] = X_feats_scaled
            df.loc[:, target] = y_target_scaled

            df['pool'] = addr  
            return get_X_y(df, features, target, n_lags)

        if verbose:
            print(f"Loading {total} pools using {num_workers} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(filter_and_process, addr) for addr in pool_addresses]
            for i, f in enumerate(futures, 1):
                X, y_cls, y_reg = f.result()
                self.X.extend(X)
                self.y_cls.extend(y_cls)
                self.y_reg.extend(y_reg)
                if verbose:
                    print(f"Processed {i}/{total}: {len(X)} samples")

        if self.X:
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y_cls = torch.tensor(np.array(self.y_cls), dtype=torch.float32)
            self.y_reg = torch.tensor(np.array(self.y_reg), dtype=torch.float32)
        else:
            d = len(features) + 1  # features + target lags (1 dimension)
            self.X = torch.empty((0, n_lags, d), dtype=torch.float32)
            self.y_cls = torch.empty((0,), dtype=torch.float32)
            self.y_reg = torch.empty((0,), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]