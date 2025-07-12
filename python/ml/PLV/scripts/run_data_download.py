"""
run_data_download.py

Script to fetch all pool addresses from a Uniswap subgraph and download their data to HDF5 using fetch_and_save_pools.
"""
import os
import argparse
from random import shuffle as random_shuffle
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--subgraph_id', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--main_pool_address', type=str, required=False, default=None)
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
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    min_rows = 100
    pool_tier = None  # 'LOW', 'MEDIUM', 'HIGH', or None for all
    hdf5_mode = 'x'  # 'w' = overwrite, 'a' = append, 'x' = fail if exists

    # ---- Fetch pool addresses ----
    print("Fetching all pool addresses from subgraph...")
    from python.utils.subgraph_utils import fetch_all_pool_addresses
    pool_addresses = fetch_all_pool_addresses(api_key, subgraph_id, pool_tier=pool_tier)
    print(f"Found {len(pool_addresses)} pools.")
    random_shuffle(pool_addresses)
    N = 1000
    pool_addresses = pool_addresses[:N]
    # Always include main_pool_address if provided
    if args.main_pool_address and args.main_pool_address not in pool_addresses:
        pool_addresses.append(args.main_pool_address)

    # ---- Download and save pool data ----
    print(f"Downloading pool data and saving to {hdf5_path} ...")
    from python.ml.PLV.data_io import fetch_and_save_pools
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

    # with pd.HDFStore(hdf5_path, "r") as store:
    #     df = store["pool_0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"]
    #     print(df[["datetime", "liquidity", "liquidity_return"]].tail(50))

    print("Done.")


if __name__ == "__main__":
    main()
