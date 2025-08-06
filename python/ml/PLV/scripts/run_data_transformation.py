"""
run_data_transformation.py

Script to load raw pool data from HDF5 and apply feature engineering transformations,
saving the transformed data to a new HDF5 file. Uses both multiprocessing (via SLURM)
and multithreading for parallel processing across multiple nodes and CPUs.

This script automatically detects SLURM environment variables to distribute pools
across multiple processes/nodes, with each process using multithreading internally.

Usage:
    # Single process:
    python run_data_transformation.py --input_hdf5 <INPUT_PATH> --output_hdf5 <OUTPUT_PATH> [--max_workers <N>]
    
    # Multi-process via SLURM:
    srun python run_data_transformation.py --input_hdf5 <INPUT_PATH> --output_hdf5 <OUTPUT_PATH> [--max_workers <N>]

Arguments:
    --input_hdf5: Path to input HDF5 file with raw pool data.
    --output_hdf5: Path to output HDF5 file for transformed data.
    --max_workers: (Optional) Number of worker threads per process (default: CPU count per task).

SLURM Environment:
    - SLURM_PROCID: Current process ID (0 to SLURM_NTASKS-1)
    - SLURM_NTASKS: Total number of processes
    - SLURM_CPUS_PER_TASK: CPUs allocated per process
"""
import os
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import h5py
import pandas as pd
import numpy as np

from python.ml.PLV.data_io import feature_engineer, get_saved_pool_addresses, load_pool_data

def process_pool(pool_address: str, input_hdf5_path: str, output_hdf5_path: str, 
                write_lock: Lock, processed_count: list, total_pools: int) -> bool:
    """
    Process a single pool: load raw data, apply feature engineering, save transformed data.
    
    Args:
        pool_address: Pool address to process
        input_hdf5_path: Path to input HDF5 file
        output_hdf5_path: Path to output HDF5 file
        write_lock: Lock for thread-safe HDF5 writing
        processed_count: List with single element to track progress
        total_pools: Total number of pools for progress tracking
        
    Returns:
        bool: True if successfully processed, False otherwise
    """
    try:
        # Load raw data
        df = load_pool_data(input_hdf5_path, pool_address)
        
        if len(df) == 0:
            with write_lock:
                processed_count[0] += 1
                print(f"[{processed_count[0]}/{total_pools}] Skipping {pool_address}: no data")
            return False
        
        # Apply feature engineering
        df_transformed = feature_engineer(df)
        
        # Save transformed data
        pool_key = f'pool_{pool_address.lower()}'
        
        with write_lock:
            with h5py.File(output_hdf5_path, 'a') as h5f:
                # Remove existing data if it exists
                if pool_key in h5f:
                    del h5f[pool_key]
                
                grp = h5f.require_group(pool_key)
                
                # Split columns by dtype
                num_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
                str_cols = df_transformed.select_dtypes(exclude=[np.number]).columns.tolist()
                
                # Save numeric data
                if num_cols:
                    grp.create_dataset('data', data=df_transformed[num_cols].to_numpy(), 
                                     compression='gzip', chunks=True)
                    dt = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset('num_columns', data=np.array(num_cols, dtype=object), dtype=dt)
                
                # Save string/object data
                if str_cols:
                    str_data = df_transformed[str_cols].astype(str).to_numpy()
                    dt = h5py.string_dtype(encoding='utf-8')
                    grp.create_dataset('strings', data=str_data, dtype=dt, 
                                     compression='gzip', chunks=True)
                    grp.create_dataset('str_columns', data=np.array(str_cols, dtype=object), dtype=dt)
            
            processed_count[0] += 1
            print(f"[{processed_count[0]}/{total_pools}] Processed {pool_address}: {len(df)} -> {len(df_transformed)} rows, {len(df_transformed.columns)} features")
        
        return True
        
    except Exception as e:
        with write_lock:
            processed_count[0] += 1
            print(f"[{processed_count[0]}/{total_pools}] Error processing {pool_address}: {str(e)}")
        return False

def main():
    """
    Main entry point for data transformation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_hdf5', type=str, required=True, 
                       help="Path to input HDF5 file with raw pool data")
    parser.add_argument('--output_hdf5', type=str, required=True, 
                       help="Path to output HDF5 file for transformed data")
    parser.add_argument('--max_workers', type=int, required=False, 
                       default=None, help="Number of worker threads per process")
    args = parser.parse_args()

    # Get SLURM task information for process distribution
    task_id = int(os.environ.get('SLURM_PROCID', '0'))
    total_tasks = int(os.environ.get('SLURM_NTASKS', '1'))
    
    print(f"[run_data_transformation.py] Process {task_id}/{total_tasks}")
    print("[run_data_transformation.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # Determine number of workers per process
    if args.max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    else:
        max_workers = args.max_workers
    
    print(f"Using max_workers={max_workers} for parallel processing within process {task_id}.")

    # Check input file exists
    if not os.path.exists(args.input_hdf5):
        print(f"Error: Input file {args.input_hdf5} does not exist.")
        sys.exit(1)

    # Get list of pool addresses from input file
    all_pool_addresses = get_saved_pool_addresses(args.input_hdf5)
    total_pools = len(all_pool_addresses)
    
    # Distribute pools across processes
    pools_per_task = total_pools // total_tasks
    remainder = total_pools % total_tasks
    
    # Calculate start and end indices for this process
    start_idx = task_id * pools_per_task + min(task_id, remainder)
    if task_id < remainder:
        end_idx = start_idx + pools_per_task + 1
    else:
        end_idx = start_idx + pools_per_task
    
    # Get subset of pools for this process
    pool_addresses = all_pool_addresses[start_idx:end_idx]
    
    print(f"Process {task_id}: Processing pools {start_idx}-{end_idx-1} ({len(pool_addresses)} pools out of {total_pools} total)")
    
    if len(pool_addresses) == 0:
        print(f"Process {task_id}: No pools assigned, exiting.")
        return

    # Create output directory if needed (only on first process)
    if task_id == 0:
        output_dir = os.path.dirname(args.output_hdf5)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize output file with metadata
        with h5py.File(args.output_hdf5, 'w') as h5f:
            meta_grp = h5f.create_group('meta')
            meta_grp.attrs['source_file'] = args.input_hdf5
            meta_grp.attrs['transformation_time'] = time.time()
            meta_grp.attrs['total_pools'] = total_pools
            meta_grp.attrs['total_processes'] = total_tasks
    
    # Wait for file initialization (simple barrier)
    time.sleep(2)

    # Process pools in parallel within this process
    write_lock = Lock()
    processed_count = [0]  # Use list for mutable reference
    successful_pools = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks for this process's subset
        futures = {
            executor.submit(process_pool, addr, args.input_hdf5, args.output_hdf5, 
                          write_lock, processed_count, len(pool_addresses)): addr
            for addr in pool_addresses
        }
        
        # Wait for completion
        for future in as_completed(futures):
            addr = futures[future]
            try:
                success = future.result()
                if success:
                    successful_pools.append(addr)
            except Exception as e:
                print(f"Process {task_id}: Unexpected error processing {addr}: {str(e)}")

    # Update metadata for this process
    with write_lock:
        with h5py.File(args.output_hdf5, 'a') as h5f:
            meta_grp = h5f['meta']
            # Store per-process statistics
            process_key = f'process_{task_id}'
            meta_grp.attrs[f'{process_key}_successful_pools'] = len(successful_pools)
            meta_grp.attrs[f'{process_key}_pool_addresses'] = ','.join(successful_pools)
            meta_grp.attrs[f'{process_key}_processing_time'] = time.time() - start_time

    elapsed_time = time.time() - start_time
    print(f"\nProcess {task_id}: Transformation completed in {elapsed_time:.2f} seconds.")
    print(f"Process {task_id}: Successfully processed {len(successful_pools)}/{len(pool_addresses)} pools.")
    
    if task_id == 0:
        print(f"Output saved to: {args.output_hdf5}")

if __name__ == "__main__":
    main()
