"""
run_data_download.py

Script to fetch all pool addresses from a Uniswap subgraph and download their data to HDF5 using fetch_and_save_pools.
"""
import os
from python.utils.subgraph_utils import fetch_all_pool_addresses
from python.ml.PLV.data_io import fetch_and_save_pools
from random import shuffle as random_shuffle

# ---- User configuration ----
api_key = "d1762c97d76a973e078c5536742bd237" 
subgraph_id = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"  
start_date = "2023-01-01"
end_date = "2025-07-01"
hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
min_rows = 100
pool_tier = None  # 'LOW', 'MEDIUM', 'HIGH', or None for all
hdf5_mode = 'w'  # 'w' = overwrite, 'a' = append, 'x' = fail if exists

# ---- Fetch pool addresses ----
print("Fetching all pool addresses from subgraph...")
pool_addresses = fetch_all_pool_addresses(api_key, subgraph_id, pool_tier=pool_tier)
print(f"Found {len(pool_addresses)} pools.")
random_shuffle(pool_addresses)

# ---- Download and save pool data ----
print(f"Downloading pool data and saving to {hdf5_path} ...")
fetch_and_save_pools(
    api_key=api_key,
    subgraph_id=subgraph_id,
    pool_addresses=pool_addresses,
    start_date=start_date,
    end_date=end_date,
    hdf5_path=hdf5_path,
    min_rows=min_rows,
    mode=hdf5_mode
)
print("Done.")
