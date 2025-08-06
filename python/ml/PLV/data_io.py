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
from python.utils.data_utils import *

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a pool DataFrame, including returns, rolling stats, and temporal features.
    All features that could leak information about future targets are properly shifted.
    
    Args:
        df (pd.DataFrame): Raw pool data.
    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    df = df.copy()
    
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
    
    # Clean up
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


def get_X_y(df: pd.DataFrame, features: List[str], target_cols: List[str], n_lags: int) -> Tuple[list, list, list, list, list]:
    """
    Convert a DataFrame to supervised learning arrays for Multi-task ZeroInflated LSTM/Transformer:
    - X: lagged features (ending at t) + target lags (ending at t-1)
    - y_cls_1, y_cls_2: classification labels (1 if target == 0, else 0) for each target
    - y_reg_1, y_reg_2: regression targets (at time t) for each target
    
    Features at time t are valid - they don't leak information about targets[t].
    
    Args:
        df: Input DataFrame
        features: List of feature column names
        target_cols: List of exactly 2 target column names [target1, target2]
        n_lags: Number of lag steps
        
    Returns:
        X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 as lists
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
        return [], [], [], [], []

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
    y_cls_1 = (y_reg_1 == 0).astype(float)
    y_cls_2 = (y_reg_2 == 0).astype(float)

    return X.tolist(), y_cls_1.tolist(), y_reg_1.tolist(), y_cls_2.tolist(), y_reg_2.tolist()


def fetch_and_save_pools(
    api_key: str,
    subgraph_id: str,
    pool_addresses: List[str],
    start_date: str,
    end_date: str,
    hdf5_path: str,
    min_rows: int = 10,
    mode: str = 'w',  # 'w' = overwrite pool, 'a' = append/update pool, 'x' = skip if exists
    fetch_mode: str = 'parallel',  # 'sequential', 'batched', or 'parallel'
    max_workers: int = 16  # Only used for parallel mode
):
    """
    Fetch raw hourly data for each pool and save to HDF5 without feature engineering.
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
                print(f"[{idx}/{total}] Saving {addr} with {n} rows (raw data)")
                # Add datetime column if not present for consistency
                if 'datetime' not in df.columns and 'periodStartUnix' in df.columns:
                    df['datetime'] = pd.to_datetime(df['periodStartUnix'], unit='s')
                
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
    targets: list,
    split_dates: dict = None,
    verbose: int = 1,
    num_workers: int = 16,
    sample_size_pct: float = 0.1,
    feature_scaler=None,
    target_reg_scalers=None,
):
    """
    Fit feature and target scalers on a random sample of the data for multi-task learning.
    Args:
        hdf5_path (str): Path to HDF5 file.
        pool_addresses (list): List of pool addresses to sample from.
        features (list): List of feature columns.
        targets (list): List of exactly 2 target column names.
        split_dates (dict): Dict with split start/end dates.
        verbose (int): Print progress if 1.
        num_workers (int): Number of threads for parallel loading.
        sample_size_pct (float): Fraction of pool_addresses to use (0 < pct <= 1).
    Returns:
        (feature_scaler, target_reg_scalers): Fitted StandardScaler objects - scalers list for 2 targets.
    """
    if len(targets) != 2:
        raise ValueError("Multi-task model requires exactly 2 targets")
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
    if target_reg_scalers is None:
        target_reg_scalers = [StandardScaler(), StandardScaler()]
    scaler_lock = Lock()

    def process_and_partial_fit(addr):
        df = data_by_pool.get(addr)
        if df is None or df.empty:
            return 0, 0, 0
        # Time filtering (use train split if available)
        if split_dates:
            start, end = split_dates.get('train_start'), split_dates.get('train_end')
            if start:
                df = df[df['datetime'] >= pd.to_datetime(start)]
            if end:
                df = df[df['datetime'] <= pd.to_datetime(end)]
        df = df.dropna(subset=features + targets)
        if df.empty:
            return 0, 0, 0
        X = df[features].values
        y1 = df[targets[0]].values.reshape(-1, 1)
        y2 = df[targets[1]].values.reshape(-1, 1)
        # Filter out non-finite rows
        mask = (np.isfinite(X).all(axis=1) & 
                np.isfinite(y1).flatten() & 
                np.isfinite(y2).flatten())
        X = X[mask]
        y1 = y1[mask]
        y2 = y2[mask]
        if X.shape[0] == 0 or y1.shape[0] == 0 or y2.shape[0] == 0:
            return 0, 0, 0
        with scaler_lock:
            feature_scaler.partial_fit(X)
            target_reg_scalers[0].partial_fit(y1)
            target_reg_scalers[1].partial_fit(y2)
        return X.shape[0], y1.shape[0], y2.shape[0]

    total_X, total_y1, total_y2 = 0, 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_and_partial_fit, addr) for addr in sample_addresses]
        for i, f in enumerate(futures, 1):
            n_x, n_y1, n_y2 = f.result()
            total_X += n_x
            total_y1 += n_y1
            total_y2 += n_y2
            if verbose:
                print(f"[fit_scalers] Processed {i}/{n_sample}: {n_x} feature, {n_y1}/{n_y2} target samples")
    if total_X == 0 or total_y1 == 0 or total_y2 == 0:
        raise ValueError("No data found for fitting scalers.")
    if verbose:
        print(f"[fit_scalers] Fitted feature scaler on {total_X} samples, target scalers on {total_y1}/{total_y2} samples")
    return feature_scaler, target_reg_scalers


class LPsDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        pool_addresses: List[str] = None,
        features: List[str] = None,
        targets: List[str] = None,  # Must be list of exactly 2 targets
        n_lags: int = 1,
        split: str = 'train',
        split_dates: dict = None,
        feature_scaler=None,
        target_reg_scalers=None,  # List of 2 scalers
        verbose: int = 0,
        num_workers: int = 16
    ):
        if targets is None or len(targets) != 2:
            raise ValueError("Must provide exactly 2 targets for multi-task learning")
        
        self.targets = targets
        self.X, self.y_cls_1, self.y_reg_1, self.y_cls_2, self.y_reg_2 = [], [], [], [], []
            
        self.split = split
        self.feature_scaler = feature_scaler
        self.target_reg_scalers = target_reg_scalers if target_reg_scalers is not None else [None, None]
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
                return [], [], [], [], []
            # Time filtering
            if start:
                df = df[df['datetime'] >= pd.to_datetime(start)]
            if end:
                df = df[df['datetime'] <= pd.to_datetime(end)]
            # Drop rows with NaNs in features or targets
            df = df.dropna(subset=features + self.targets)
            if df.empty:
                return [], [], [], [], []
            
            # Scale features independently per pool
            X_feats = df[features].values
            # Ensure no non-finite values in features
            if not np.all(np.isfinite(X_feats)):
                if verbose:
                    print(f"[{addr}] Skipping due to bad feature values")
                return [], [], [], [], []

            # Use passed-in scaler if provided, else fit per-pool
            if self.feature_scaler is not None:
                X_feats_scaled = self.feature_scaler.transform(X_feats)
            else:
                feature_scaler = StandardScaler()
                X_feats_scaled = feature_scaler.fit_transform(X_feats)
            df.loc[:, features] = X_feats_scaled

            for i, target in enumerate(self.targets):
                y_target = df[target].values.reshape(-1, 1)
                # Ensure no non-finite values in target
                if not np.all(np.isfinite(y_target)):
                    if verbose:
                        print(f"[{addr}] Skipping due to bad target values in {target}")
                    return [], [], [], [], []
                
                # Use passed-in scaler if provided, else fit per-pool
                if self.target_reg_scalers[i] is not None:
                    y_target_scaled = self.target_reg_scalers[i].transform(y_target).flatten()
                else:
                    target_scaler = StandardScaler()
                    y_target_scaled = target_scaler.fit_transform(y_target).flatten()
                df.loc[:, target] = y_target_scaled
            
            df['pool'] = addr
            return get_X_y(df, features, self.targets, n_lags)
            

        if verbose:
            print(f"Loading {total} pools using {num_workers} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(filter_and_process, addr) for addr in pool_addresses]
            for i, f in enumerate(futures, 1):
                X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = f.result()
                self.X.extend(X)
                self.y_cls_1.extend(y_cls_1)
                self.y_reg_1.extend(y_reg_1)
                self.y_cls_2.extend(y_cls_2)
                self.y_reg_2.extend(y_reg_2)
                if verbose:
                    print(f"Processed {i}/{total}: {len(X)} samples")

        if self.X:
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y_cls_1 = torch.tensor(np.array(self.y_cls_1), dtype=torch.float32)
            self.y_reg_1 = torch.tensor(np.array(self.y_reg_1), dtype=torch.float32)
            self.y_cls_2 = torch.tensor(np.array(self.y_cls_2), dtype=torch.float32)
            self.y_reg_2 = torch.tensor(np.array(self.y_reg_2), dtype=torch.float32)
        else:
            d = len(features) + len(self.targets)  # features + target lags
            self.X = torch.empty((0, n_lags, d), dtype=torch.float32)
            self.y_cls_1 = torch.empty((0,), dtype=torch.float32)
            self.y_reg_1 = torch.empty((0,), dtype=torch.float32) 
            self.y_cls_2 = torch.empty((0,), dtype=torch.float32)
            self.y_reg_2 = torch.empty((0,), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], 
               self.y_cls_1[idx], self.y_reg_1[idx],
               self.y_cls_2[idx], self.y_reg_2[idx])