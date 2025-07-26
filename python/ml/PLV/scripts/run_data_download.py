"""
run_data_download.py

Script to fetch all pool addresses from a Uniswap subgraph and download their data to HDF5 using fetch_and_save_pools.

Usage:
    python run_data_download.py --api_key <API_KEY> --subgraph_id <SUBGRAPH_ID> --start_date <YYYY-MM-DD> --end_date <YYYY-MM-DD> [--main_pool_address <ADDR>] [--n_pools <N>]

Arguments:
    --api_key: The Graph API key for authentication.
    --subgraph_id: The Uniswap subgraph ID to query.
    --start_date: Start date for data download (YYYY-MM-DD).
    --end_date: End date for data download (YYYY-MM-DD).
    --main_pool_address: (Optional) Always include this pool address.
    --n_pools: (Optional) Number of pools to download (default: 1000).
"""
import os
import argparse
from random import shuffle as random_shuffle
import sys

from python.utils.subgraph_utils import fetch_all_pool_addresses
from python.ml.PLV.data_io import fetch_and_save_pools

def main():
    """
    Main entry point for data download script.
    Parses arguments, fetches pool addresses, and downloads pool data to HDF5.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True, help="The Graph API key.")
    parser.add_argument('--subgraph_id', type=str, required=True, help="Uniswap subgraph ID.")
    parser.add_argument('--start_date', type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument('--end_date', type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument('--main_pool_address', type=str, required=False, default=None, help="Always include this pool address.")
    parser.add_argument('--n_pools', type=int, required=False, default=1000, help="Number of pools to download.")
    args = parser.parse_args()

    print("[run_data_download.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # ---- User configuration ----
    api_key = args.api_key
    subgraph_id = args.subgraph_id
    start_date = args.start_date
    end_date = args.end_date
    # hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    min_rows = 50
    pool_tier = None  # 'LOW', 'MEDIUM', 'HIGH', or None for all
    hdf5_mode = 'w'  # 'w' = overwrite, 'a' = append, 'x' = fail if exists

    # ---- Fetch pool addresses ----
    print("Fetching all pool addresses from subgraph...")
    pool_addresses = fetch_all_pool_addresses(api_key, subgraph_id, pool_tier=pool_tier)
    print(f"Found {len(pool_addresses)} pools.")
    random_shuffle(pool_addresses)
    N = args.n_pools
    pool_addresses = pool_addresses[:N]
    # Always include main_pool_address if provided
    if args.main_pool_address and args.main_pool_address not in pool_addresses:
        pool_addresses.append(args.main_pool_address)

    # ---- Download and save pool data ----
    print(f"Downloading pool data and saving to {hdf5_path} ...")
    # Determine number of workers (CPUs/threads)
    num_workers = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
    print(f"Using max_workers={num_workers} for parallel fetching.")
    fetch_and_save_pools(
        api_key=api_key,
        subgraph_id=subgraph_id,
        pool_addresses=pool_addresses,
        start_date=start_date,
        end_date=end_date,
        hdf5_path=hdf5_path,
        min_rows=min_rows,
        mode=hdf5_mode,
        max_workers=num_workers
    )

    print("Done.")

if __name__ == "__main__":
    main()
