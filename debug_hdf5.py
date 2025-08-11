#!/usr/bin/env python3
"""
Quick script to debug what's in the HDF5 file
"""
import h5py
import sys

def check_hdf5_contents(hdf5_path):
    try:
        print(f"Opening HDF5 file: {hdf5_path}")
        with h5py.File(hdf5_path, 'r') as h5f:
            print(f"Top-level keys: {list(h5f.keys())}")
            
            # Get pool keys
            pool_keys = [k for k in h5f.keys() if k.startswith('pool_')]
            print(f"Number of pools: {len(pool_keys)}")
            
            if pool_keys:
                sample_pool = pool_keys[0]
                print(f"\nChecking sample pool: {sample_pool}")
                pool_group = h5f[sample_pool]
                print(f"Pool group keys: {list(pool_group.keys())}")
                
                if 'num_columns' in pool_group:
                    num_columns = [col.decode() for col in pool_group['num_columns'][:]]
                    print(f"\nNumeric columns ({len(num_columns)}):")
                    for i, col in enumerate(num_columns):
                        print(f"  {i+1:2d}. {col}")
                        
                if 'str_columns' in pool_group:
                    str_columns = [col.decode() for col in pool_group['str_columns'][:]]
                    print(f"\nString columns ({len(str_columns)}):")
                    for i, col in enumerate(str_columns):
                        print(f"  {i+1:2d}. {col}")
                        
                if 'data' in pool_group:
                    data_shape = pool_group['data'].shape
                    print(f"\nData shape: {data_shape}")
                    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    hdf5_path = "/p/scratch/training2529/uniswap_pools_data_transformed.h5"
    check_hdf5_contents(hdf5_path)
