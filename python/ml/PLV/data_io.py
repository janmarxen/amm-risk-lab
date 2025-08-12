"""
data_io.py

Simplified utility functions and PyTorch Dataset for loading, engineering, and preparing Uniswap V3 pool data for ML models.
Only supports standard LPsDataset with dataset sharding for distributed training.
"""
import pandas as pd
import time
from typing import List, Dict
import numpy as np
import h5py
import gc

from sklearn.preprocessing import StandardScaler
import random

import torch
from torch.utils.data import Dataset

import concurrent.futures
from threading import Lock

from python.utils.subgraph_utils import fetch_pool_hourly_data, fetch_pools_hourly_data_batched, fetch_pools_hourly_data_batched_parallel
from python.utils.data_utils import get_X_y, get_X_y_hybrid, dropna_features_targets

def write_pools_to_hdf5(
    h5f,
    pool_data_dict: Dict[str, pd.DataFrame],
    hdf5_path: str,
    min_rows: int = 10,
    mode: str = 'a',  # 'w' = overwrite pool, 'a' = append/update pool, 'x' = skip if exists
    data_description: str = "data"  # Description for logging (e.g., "raw data", "transformed data")
) -> List[str]:
    """
    Write pool data from a dictionary to HDF5 file.
    
    Args:
        h5f: Open HDF5 file handle.
        pool_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping pool addresses to DataFrames.
        hdf5_path (str): Path to HDF5 file (for logging).
        min_rows (int): Minimum number of rows required to save pool.
        mode (str): 'w' (overwrite), 'a' (append/update), 'x' (skip if exists).
        data_description (str): Description for logging.
    
    Returns:
        List[str]: List of successfully saved pool addresses (lowercase).
    """
    fetched = []
    pool_addresses = list(pool_data_dict.keys())
    total = len(pool_addresses)
    
    for idx, addr in enumerate(pool_addresses, 1):
        pool_key = f'pool_{addr.lower()}'
        if mode == 'x' and pool_key in h5f:
            print(f"Skipping existing pool {addr} ({idx}/{total})")
            continue
        
        df = pool_data_dict.get(addr, pd.DataFrame())
        n = len(df)
        if df is not None and n >= min_rows:
            # Separate numeric and string columns for efficient storage
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            string_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            grp = h5f.require_group(pool_key)
            
            # Store numeric data
            if numeric_cols:
                numeric_data = df[numeric_cols].values
                if 'data' in grp:
                    del grp['data']
                if 'num_columns' in grp:
                    del grp['num_columns']
                grp.create_dataset('data', data=numeric_data, compression='gzip')
                grp.create_dataset('num_columns', data=[col.encode() for col in numeric_cols])
            
            # Store string data separately
            if string_cols:
                string_data = df[string_cols].values.astype('S')
                if 'strings' in grp:
                    del grp['strings']
                if 'str_columns' in grp:
                    del grp['str_columns']
                grp.create_dataset('strings', data=string_data, compression='gzip')
                grp.create_dataset('str_columns', data=[col.encode() for col in string_cols])
            
            fetched.append(addr.lower())
            print(f"Saved pool {addr} with {n} rows of {data_description} ({idx}/{total})")
        else:
            print(f"Skipped pool {addr}: insufficient data ({n} < {min_rows} rows) ({idx}/{total})")
        
        # Save metadata
        meta_grp = h5f.require_group('meta')
        meta_grp.attrs['pool_addresses'] = ','.join(fetched)
        meta_grp.attrs['fetch_time'] = time.time()
    
    return fetched


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
    # Open HDF5 file in append mode
    with h5py.File(hdf5_path, 'a') as h5f:
        # Fetch all pools according to fetch_mode
        if fetch_mode == 'sequential':
            pool_data_dict = {}
            for addr in pool_addresses:
                pool_data_dict[addr] = fetch_pool_hourly_data(api_key, subgraph_id, addr, start_date, end_date)
        elif fetch_mode == 'batched':
            pool_data_dict = fetch_pools_hourly_data_batched(api_key, subgraph_id, pool_addresses, start_date, end_date)
        else:            
            pool_data_dict = fetch_pools_hourly_data_batched_parallel(
                api_key, subgraph_id, pool_addresses, start_date, end_date, max_workers=max_workers
            )
        
        # Use the refactored write function
        fetched = write_pools_to_hdf5(
            h5f, pool_data_dict, hdf5_path, 
            min_rows=min_rows, mode=mode, data_description="raw data"
        )
    
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
            raise KeyError(f"Pool {pool_address} not found in HDF5 file")
        grp = h5f[pool_key]
        dfs = []
        # Numeric columns
        if 'data' in grp and 'num_columns' in grp:
            num_data = grp['data'][:]
            num_columns = [col.decode() for col in grp['num_columns'][:]]
            dfs.append(pd.DataFrame(num_data, columns=num_columns))
        # String columns
        if 'strings' in grp and 'str_columns' in grp:
            str_data = grp['strings'][:]
            str_columns = [col.decode() for col in grp['str_columns'][:]]
            str_df = pd.DataFrame(str_data, columns=str_columns)
            # Decode bytes to strings
            for col in str_columns:
                str_df[col] = str_df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
            dfs.append(str_df)
        if dfs:
            return pd.concat(dfs, axis=1)
        else:
            raise ValueError(f"No data found for pool {pool_address}")

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
            try:
                pool_key = f'pool_{addr.lower()}'
                if pool_key in h5f:
                    pool_dict[addr] = load_pool_data(hdf5_path, addr)
            except (KeyError, ValueError):
                continue  # Skip pools that can't be loaded
    return pool_dict


def stream_pool_data(hdf5_path: str, pool_addresses: list):
    """
    Generator that yields (address, DataFrame) tuples one at a time for memory efficiency.
    Use this when you need to process pools sequentially without keeping all in memory.
    
    Args:
        hdf5_path (str): Path to HDF5 file.
        pool_addresses (list): List of pool addresses to stream.
    
    Yields:
        Tuple[str, pd.DataFrame]: (pool_address, dataframe) pairs.
    """
    with h5py.File(hdf5_path, 'r') as h5f:
        for addr in pool_addresses:
            try:
                df = load_pool_data(hdf5_path, addr)
                yield addr, df
            except (KeyError, ValueError):
                continue  # Skip pools that can't be loaded


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
                addr = key[5:]  # Remove 'pool_' prefix
                try:
                    pool_dict[addr] = load_pool_data(hdf5_path, addr)
                except (KeyError, ValueError):
                    continue  # Skip pools that can't be loaded
    return pool_dict

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
            addrs_str = h5f['meta'].attrs.get('pool_addresses', '')
            if addrs_str:
                return addrs_str.split(',')
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
            start_date = split_dates.get('train_start')
            end_date = split_dates.get('train_end')
            if start_date and end_date and 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()
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
    """
    PyTorch Dataset for Liquidity Pool (LP) data with multi-task learning support.
    Loads all specified pools into memory for fast access during training.
    """
    def __init__(
        self,
        hdf5_path: str,
        pool_addresses: List[str] = None,
        features: List[str] = None,
        targets: List[str] = None,        
        n_lags: int = 1,
        split: str = 'train',
        split_dates: dict = None,
        feature_scaler=None,
        target_reg_scalers=None,        
        verbose: int = 0,
        num_workers: int = 16
    ):
        """
        Initialize LPsDataset.
        Args:
            hdf5_path (str): Path to HDF5 file with pool data.
            pool_addresses (List[str]): List of pool addresses to load.
            features (List[str]): List of feature column names.
            targets (List[str]): List of exactly 2 target column names.
            n_lags (int): Number of time lags for sequence modeling.
            split (str): Data split ('train', 'val', 'test').
            split_dates (dict): Dictionary with date ranges for each split.
            feature_scaler: Fitted feature scaler (StandardScaler).
            target_reg_scalers: List of 2 fitted target scalers.
            verbose (int): Verbosity level.
            num_workers (int): Number of threads for parallel data loading.
        """
        if targets is None or len(targets) != 2:
            raise ValueError("Multi-task model requires exactly 2 targets")
        
        self.hdf5_path = hdf5_path
        self.features = features
        self.targets = targets
        self.n_lags = n_lags
        self.split = split
        self.split_dates = split_dates
        self.feature_scaler = feature_scaler
        self.target_reg_scalers = target_reg_scalers if target_reg_scalers is not None else [None, None]
        self.verbose = verbose
        
        # Get pool addresses
        if pool_addresses is None:
            pool_addresses = get_saved_pool_addresses(hdf5_path)
        self.pool_addresses = pool_addresses
        
        # Load all data into memory
        if verbose:
            print(f"[LPsDataset] Loading {len(pool_addresses)} pools into memory...")
        
        self.X = []
        self.y_cls_1 = []
        self.y_reg_1 = []
        self.y_cls_2 = []
        self.y_reg_2 = []
        self._load_all_data(num_workers)
        
        if verbose:
            print(f"[LPsDataset] Loaded {len(self.X)} total samples from {len(pool_addresses)} pools")

    def _load_all_data(self, num_workers):
        """Load all pool data into memory with parallel processing."""
        def load_pool(pool_addr):
            try:
                df = load_pool_data(self.hdf5_path, pool_addr)
                
                # Apply time filtering based on split
                if self.split_dates:
                    start_date, end_date = self._get_split_dates()
                    if start_date and end_date and 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()
                
                # Drop missing values
                df = df.dropna(subset=self.features + self.targets)
                if df.empty:
                    return [], [], [], [], []
                
                # Apply scaling before creating lags
                if self.feature_scaler is not None:
                    # Use provided scalers
                    df[self.features] = self.feature_scaler.transform(df[self.features])
                    if self.target_reg_scalers[0] is not None:
                        df[[self.targets[0]]] = self.target_reg_scalers[0].transform(df[[self.targets[0]]])
                    if self.target_reg_scalers[1] is not None:
                        df[[self.targets[1]]] = self.target_reg_scalers[1].transform(df[[self.targets[1]]])
                else:
                    # Create temporary scalers and fit_transform
                    temp_feature_scaler = StandardScaler()
                    df[self.features] = temp_feature_scaler.fit_transform(df[self.features])
                    
                    temp_target_scaler_1 = StandardScaler()
                    df[[self.targets[0]]] = temp_target_scaler_1.fit_transform(df[[self.targets[0]]])
                    
                    temp_target_scaler_2 = StandardScaler()
                    df[[self.targets[1]]] = temp_target_scaler_2.fit_transform(df[[self.targets[1]]])
                
                # Get sequences with lags
                X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = get_X_y(
                    df=df,
                    features=self.features,
                    target_cols=self.targets,
                    n_lags=self.n_lags
                )
                
                if X is None or len(X) == 0:
                    return [], [], [], [], []
                
                return X, y_cls_1, y_reg_1, y_cls_2, y_reg_2
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load pool {pool_addr}: {e}")
                return [], [], [], [], []
        
        # Load pools in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(load_pool, addr) for addr in self.pool_addresses]
            
            for i, future in enumerate(futures, 1):
                X_pool, y_cls_1_pool, y_reg_1_pool, y_cls_2_pool, y_reg_2_pool = future.result()
                if len(X_pool) > 0:
                    # Convert to lists if they're arrays
                    if isinstance(X_pool, np.ndarray):
                        X_pool = X_pool.tolist()
                    if isinstance(y_cls_1_pool, np.ndarray):
                        y_cls_1_pool = y_cls_1_pool.tolist()
                    if isinstance(y_reg_1_pool, np.ndarray):
                        y_reg_1_pool = y_reg_1_pool.tolist()
                    if isinstance(y_cls_2_pool, np.ndarray):
                        y_cls_2_pool = y_cls_2_pool.tolist()
                    if isinstance(y_reg_2_pool, np.ndarray):
                        y_reg_2_pool = y_reg_2_pool.tolist()
                    
                    self.X.extend(X_pool)
                    self.y_cls_1.extend(y_cls_1_pool)
                    self.y_reg_1.extend(y_reg_1_pool)
                    self.y_cls_2.extend(y_cls_2_pool)
                    self.y_reg_2.extend(y_reg_2_pool)
                
                if self.verbose and i % 100 == 0:
                    print(f"[LPsDataset] Processed {i}/{len(self.pool_addresses)} pools, {len(self.X)} samples so far")
        
        # Convert to tensors
        if len(self.X) > 0:
            self.X = torch.FloatTensor(np.array(self.X))
            self.y_cls_1 = torch.FloatTensor(np.array(self.y_cls_1))
            self.y_reg_1 = torch.FloatTensor(np.array(self.y_reg_1))
            self.y_cls_2 = torch.FloatTensor(np.array(self.y_cls_2))
            self.y_reg_2 = torch.FloatTensor(np.array(self.y_reg_2))
        else:
            # Empty dataset fallback
            input_size = len(self.features)
            self.X = torch.zeros(0, self.n_lags, input_size)
            self.y_cls_1 = torch.zeros(0)
            self.y_reg_1 = torch.zeros(0)
            self.y_cls_2 = torch.zeros(0)
            self.y_reg_2 = torch.zeros(0)

    def _get_split_dates(self):
        """Get start and end dates for the current split."""
        if self.split_dates is None:
            return None, None
        
        if self.split == 'train':
            return self.split_dates.get('train_start'), self.split_dates.get('train_end')
        elif self.split == 'val':
            return self.split_dates.get('val_start'), self.split_dates.get('val_end')
        elif self.split == 'test':
            return self.split_dates.get('test_start'), self.split_dates.get('test_end')
        else:
            return None, None

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.X[idx], self.y_cls_1[idx], self.y_reg_1[idx], self.y_cls_2[idx], self.y_reg_2[idx]


class HybridLPsDataset(Dataset):
    """
    PyTorch Dataset for Hybrid Multi-task learning:
    - Task 1 (liquidity_return): Zero-inflated (classification + regression)
    - Task 2 (volume_return): Standard regression only
    
    Loads all specified pools into memory for fast access during training.
    """
    def __init__(
        self,
        hdf5_path: str,
        pool_addresses: List[str] = None,
        features: List[str] = None,
        targets: List[str] = None,        
        n_lags: int = 1,
        split: str = 'train',
        split_dates: dict = None,
        feature_scaler=None,
        target_reg_scalers=None,        
        verbose: int = 0,
        num_workers: int = 16
    ):
        """
        Initialize HybridLPsDataset.
        Args:
            hdf5_path (str): Path to HDF5 file with pool data.
            pool_addresses (List[str]): List of pool addresses to load.
            features (List[str]): List of feature column names.
            targets (List[str]): List of exactly 2 target column names [liquidity_return, volume_return].
            n_lags (int): Number of time lags for sequence modeling.
            split (str): Data split ('train', 'val', 'test').
            split_dates (dict): Dictionary with date ranges for each split.
            feature_scaler: Fitted feature scaler (StandardScaler).
            target_reg_scalers: List of 2 fitted target scalers [liquidity_scaler, volume_scaler].
            verbose (int): Verbosity level.
            num_workers (int): Number of parallel workers.
        """
        self.hdf5_path = hdf5_path
        self.pool_addresses = pool_addresses or []
        self.features = features or []
        self.targets = targets or []
        self.n_lags = n_lags
        self.split = split
        self.split_dates = split_dates or {}
        self.feature_scaler = feature_scaler
        self.target_reg_scalers = target_reg_scalers if target_reg_scalers is not None else [None, None]
        self.verbose = verbose
        self.num_workers = num_workers
        
        # Initialize data containers
        self.X = []
        self.y_cls_1 = []  # Only for task 1 (liquidity_return)
        self.y_reg_1 = []  # Task 1 regression
        self.y_reg_2 = []  # Task 2 regression (volume_return)
        
        self._load_data()
        
        # Convert to tensors
        if self.X:
            self.X = torch.FloatTensor(np.array(self.X))
            self.y_cls_1 = torch.FloatTensor(np.array(self.y_cls_1))
            self.y_reg_1 = torch.FloatTensor(np.array(self.y_reg_1))
            self.y_reg_2 = torch.FloatTensor(np.array(self.y_reg_2))

    def _load_data(self):
        """Load and process data from all pools."""
        def process_pool(pool_address):
            try:
                # Load pool data
                df = load_pool_data(self.hdf5_path, pool_address)
                
                # Apply date filtering
                if self.split_dates:
                    start_date, end_date = self._get_split_dates()
                    if start_date and end_date and 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()
                
                if len(df) < self.n_lags + 1:
                    return None
                
                # Apply scaling
                df = df.copy()
                if self.features:
                    # Feature scaling
                    if self.feature_scaler is not None:
                        df[self.features] = self.feature_scaler.transform(df[self.features])
                    
                    # Target scaling
                    if self.target_reg_scalers[0] is not None:
                        df[[self.targets[0]]] = self.target_reg_scalers[0].transform(df[[self.targets[0]]])
                    if self.target_reg_scalers[1] is not None:
                        df[[self.targets[1]]] = self.target_reg_scalers[1].transform(df[[self.targets[1]]])
                else:
                    # Fit temporary scalers for this pool
                    temp_feature_scaler = StandardScaler()
                    df[self.features] = temp_feature_scaler.fit_transform(df[self.features])
                    
                    temp_target_scaler_1 = StandardScaler()
                    df[[self.targets[0]]] = temp_target_scaler_1.fit_transform(df[[self.targets[0]]])
                    
                    temp_target_scaler_2 = StandardScaler()
                    df[[self.targets[1]]] = temp_target_scaler_2.fit_transform(df[[self.targets[1]]])
                
                # Create sequences using hybrid function
                X, y_cls_1, y_reg_1, y_reg_2 = get_X_y_hybrid(
                    df, self.features, self.targets, self.n_lags
                )
                
                if not X:
                    return None
                    
                return X, y_cls_1, y_reg_1, y_reg_2
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[HybridLPsDataset] Error processing pool {pool_address}: {e}")
                return None

        # Load pools in parallel
        if self.verbose >= 1:
            print(f"[HybridLPsDataset] Loading {len(self.pool_addresses)} pools into memory...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_pool, pool_addr) for pool_addr in self.pool_addresses]
            
            processed_count = 0
            total_samples = 0
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result is not None:
                    X_pool, y_cls_1_pool, y_reg_1_pool, y_reg_2_pool = result
                    
                    # Convert to lists if they're numpy arrays
                    if isinstance(X_pool, np.ndarray):
                        X_pool = X_pool.tolist()
                    if isinstance(y_cls_1_pool, np.ndarray):
                        y_cls_1_pool = y_cls_1_pool.tolist()
                    if isinstance(y_reg_1_pool, np.ndarray):
                        y_reg_1_pool = y_reg_1_pool.tolist()
                    if isinstance(y_reg_2_pool, np.ndarray):
                        y_reg_2_pool = y_reg_2_pool.tolist()
                    
                    # Extend the main containers
                    self.X.extend(X_pool)
                    self.y_cls_1.extend(y_cls_1_pool)
                    self.y_reg_1.extend(y_reg_1_pool)
                    self.y_reg_2.extend(y_reg_2_pool)
                    
                    total_samples += len(X_pool)
                
                processed_count += 1
                if self.verbose >= 1 and processed_count % 100 == 0:
                    print(f"[HybridLPsDataset] Processed {processed_count}/{len(self.pool_addresses)} pools, {total_samples} samples so far")
        
        if self.verbose >= 1:
            print(f"[HybridLPsDataset] Loaded {total_samples} total samples from {len(self.pool_addresses)} pools")

    def _get_split_dates(self):
        """Get start and end dates for the current split."""
        if self.split_dates is None:
            return None, None
        if self.split == 'train':
            return self.split_dates.get('train_start'), self.split_dates.get('train_end')
        elif self.split == 'val':
            return self.split_dates.get('val_start'), self.split_dates.get('val_end')
        elif self.split == 'test':
            return self.split_dates.get('test_start'), self.split_dates.get('test_end')
        else:
            return None, None

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get a sample by index for hybrid architecture."""
        return self.X[idx], self.y_cls_1[idx], self.y_reg_1[idx], self.y_reg_2[idx]
