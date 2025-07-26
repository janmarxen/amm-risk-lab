"""
plv_data_io.py

Utility functions and PyTorch Dataset for loading, engineering, and preparing Uniswap V3 pool data for ML models.
"""
import pandas as pd
import time
from typing import List, Dict
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

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
    df['price'] = df['price'].astype(float)
    df['liquidity'] = df['liquidity'].astype(float)
    df['volumeUSD'] = df['volumeUSD'].astype(float)
    # Returns
    df['price_return'] = df['price'].pct_change()
    df['liquidity_return'] = df['liquidity'].pct_change()
    df['volume_return'] = df['volumeUSD'].pct_change()
    # Outlier removal
    def remove_outliers_iqr(series, k=3.0):
        nonzero = series[series != 0]
        q1 = nonzero.quantile(0.25)
        q3 = nonzero.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        filtered = series.where((series >= lower) & (series <= upper))
        filtered = filtered.interpolate(method='linear', limit_direction='both')
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

def get_X_y(df: pd.DataFrame, features: List[str], target_col: str, n_lags: int):
    """
    Convert a DataFrame to supervised learning arrays for LSTM:
    - X: lagged feature windows (including present values at i)
    - y_cls: zero-class labels
    - y_reg: regression targets
    For the target, only lags up to i-1 are included in X (not the present value).
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (List[str]): List of feature columns.
        target_col (str): Target column name.
        n_lags (int): Number of lag steps.
    Returns:
        tuple: (X, y_cls, y_reg) lists for supervised learning.
    """
    X, y_cls, y_reg = [], [], []
    for i in range(n_lags, len(df)):
        # Features: use values from i-n_lags+1 to i (inclusive, present included)
        X_feats = df[features].iloc[i - n_lags + 1:i+1].values
        # Target lags: use values from i-n_lags to i-1 (present excluded)
        target_lags = df[target_col].iloc[i - n_lags:i].values.reshape(-1, 1)
        # Concatenate features and target lags along the last axis
        X_window = np.concatenate([X_feats, target_lags], axis=1)
        target = df[target_col].iloc[i]
        # Remove samples with any NaN or inf in features or target
        if not (np.all(np.isfinite(X_window)) and np.isfinite(target)):
            continue
        X.append(X_window)
        y_cls.append(1 if target == 0 else 0)
        y_reg.append(target)
    return X, y_cls, y_reg

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
    import h5py
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

def load_all_pools_in_memory(hdf5_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all pools' data from HDF5 as a dict of address -> DataFrame, kept in memory, using h5py.
    Args:
        hdf5_path (str): Path to HDF5 file.
    Returns:
        Dict[str, pd.DataFrame]: Mapping pool address to DataFrame.
    """
    import h5py
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
                self.X = torch.empty((0, n_lags, len(features)), dtype=torch.float32)
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
    import h5py
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

class LPsDataset(Dataset):
    """
    PyTorch Dataset for LSTM pretraining/fine-tuning on multiple pools from HDF5.
    Each item is a tuple (X, y_cls, y_reg) for supervised learning.
    Allows flexible split: 'train', 'val', or 'test' using split_dates dict.
    split_dates include 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'.
    Args:
        hdf5_path (str): Path to HDF5 file.
        pool_addresses (list, optional): List of pool addresses to load. If None, loads all saved pool addresses.
        features (list, optional): List of feature columns to use.
        target (str, optional): Name of target column.
        n_lags (int, optional): Number of lag steps for LSTM.
        split (str, optional): 'train', 'val', or 'test'.
        split_dates (dict, optional): Dict with keys 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end' for splitting.
        verbose (int, optional): Print progress if 1.
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
            split_dates: Dict with keys 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end' for splitting.
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
        if self.X:
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)  # Convert list of arrays to single ndarray first
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
