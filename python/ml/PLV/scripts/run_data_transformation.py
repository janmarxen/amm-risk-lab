"""
run_data_transformation.py

Script to load raw pool data from HDF5 and apply feature engineering transformations,
saving the transformed data to a new HDF5 file. Uses MPI for distributed processing
with atomic writes to prevent HDF5 corruption.

Usage:
    # Single process:
    python run_data_transformation.py --input_hdf5 <INPUT_PATH> --output_hdf5 <OUTPUT_PATH> [--max_workers <N>]
    
    # Multi-process via MPI:
    mpirun -n <N_PROCESSES> python run_data_transformation.py --input_hdf5 <INPUT_PATH> --output_hdf5 <OUTPUT_PATH> [--max_workers <N>]

Arguments:
    --input_hdf5: Path to input HDF5 file with raw pool data.
    --output_hdf5: Path to output HDF5 file for transformed data.
    --max_workers: (Optional) Number of worker threads per process (default: CPU count per task).
"""
import os
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import pandas as pd
import numpy as np

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

from python.ml.PLV.data_io import get_saved_pool_addresses, load_pool_data, write_pools_to_hdf5
from python.utils.data_utils import feature_engineer

def process_pool_in_memory(pool_address: str, input_hdf5_path: str) -> dict:
    """
    Process a single pool: load raw data, apply feature engineering, return transformed data.
    
    Args:
        pool_address: Pool address to process
        input_hdf5_path: Path to input HDF5 file
        
    Returns:
        dict: Dictionary with processed data or None if failed
    """
    try:
        # Load raw data
        df = load_pool_data(input_hdf5_path, pool_address)
        
        if len(df) == 0:
            return {'status': 'skipped', 'reason': 'no data', 'pool_address': pool_address}
        
        # Apply feature engineering
        df_transformed = feature_engineer(df)
        
        # Split columns by dtype for storage
        num_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
        str_cols = df_transformed.select_dtypes(exclude=[np.number]).columns.tolist()
        
        result = {
            'status': 'success',
            'pool_address': pool_address,
            'num_cols': num_cols,
            'str_cols': str_cols,
            'original_rows': len(df),
            'transformed_rows': len(df_transformed),
            'num_features': len(df_transformed.columns)
        }
        
        # Store complete dataframe information
        result['data'] = df_transformed  # Store the full dataframe
        result['index'] = df_transformed.index  # Store the index separately
            
        return result
        
    except Exception as e:
        return {'status': 'error', 'pool_address': pool_address, 'error': str(e)}


def write_processed_pools_to_hdf5(pool_results: list, output_hdf5_path: str) -> int:
    """
    Atomically write processed pool data to HDF5 file using the shared write function.
    
    Args:
        pool_results: List of processed pool results
        output_hdf5_path: Path to output HDF5 file
        process_id: Process ID for logging
        
    Returns:
        int: Number of successfully written pools
    """
    successful_writes = 0
    
    # Convert results to the format expected by write_pools_to_hdf5
    pool_data_dict = {}
    pool_addresses = []
    
    for result in pool_results:
        if result['status'] == 'success':
            pool_address = result['pool_address']
            df = result['data']
            
            # Reset index to make datetime a regular column if it's the index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            pool_data_dict[pool_address] = df
            pool_addresses.append(pool_address)
    
    # Use the shared write function
    # PROBLEM IS THAT MAYBE ONE PROCESS IS OVERWRITING OTHER PROCESSES' DATA
    # TEST WITH ONE PROCESS?
    with h5py.File(output_hdf5_path, 'a') as h5f:
        fetched = write_pools_to_hdf5(
            h5f, pool_data_dict, output_hdf5_path,
            min_rows=0, mode='a', data_description="transformed data"
        )
        successful_writes = len(fetched)
    
    return successful_writes

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

    # Initialize MPI if available, otherwise use single process
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
        comm = None
    
    print(f"[run_data_transformation.py] Process {rank}/{size-1}")
    print("[run_data_transformation.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # Determine number of workers per process
    if args.max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    else:
        max_workers = args.max_workers
    
    # Limit threads to prevent HDF5 contention
    max_workers = min(max_workers, 32)
    print(f"Using max_workers={max_workers} for parallel processing within process {rank}.")

    # Check input file exists
    if not os.path.exists(args.input_hdf5):
        print(f"Error: Input file {args.input_hdf5} does not exist.")
        sys.exit(1)

    # Get list of pool addresses from input file (all processes read this)
    all_pool_addresses = get_saved_pool_addresses(args.input_hdf5)
    total_pools = len(all_pool_addresses)
    print(f"Total pools found in input file: {total_pools}")
    
    # Distribute pools across processes
    pools_per_task = total_pools // size
    remainder = total_pools % size
    
    # Calculate start and end indices for this process
    start_idx = rank * pools_per_task + min(rank, remainder)
    if rank < remainder:
        end_idx = start_idx + pools_per_task + 1
    else:
        end_idx = start_idx + pools_per_task
    
    # Get subset of pools for this process
    pool_addresses = all_pool_addresses[start_idx:end_idx]
    
    print(f"Process {rank}: Processing pools {start_idx}-{end_idx-1} ({len(pool_addresses)} pools out of {total_pools} total)")
    
    if len(pool_addresses) == 0:
        print(f"Process {rank}: No pools assigned, exiting.")
        if MPI_AVAILABLE:
            comm.Barrier()  # Wait for other processes
        return
    
    # MPI Barrier to ensure file is created before processing
    if MPI_AVAILABLE:
        comm.Barrier()
    else:
        time.sleep(2)

    # Process pools in parallel within this process (keep in memory)
    pool_results = []
    processed_count = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks for this process's subset
        futures = {
            executor.submit(process_pool_in_memory, addr, args.input_hdf5): addr
            for addr in pool_addresses
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            addr = futures[future]
            try:
                result = future.result()
                pool_results.append(result)
                processed_count += 1
                
                if result['status'] == 'success':
                    print(f"[{processed_count}/{len(pool_addresses)}] Processed {result['pool_address']}: {result['original_rows']} -> {result['transformed_rows']} rows, {result['num_features']} features")
                elif result['status'] == 'skipped':
                    print(f"[{processed_count}/{len(pool_addresses)}] Skipped {result['pool_address']}: {result['reason']}")
                else:  # error
                    print(f"[{processed_count}/{len(pool_addresses)}] Error processing {result['pool_address']}: {result['error']}")
                    
            except Exception as e:
                processed_count += 1
                print(f"Process {rank}: Unexpected error processing {addr}: {str(e)}")
                pool_results.append({'status': 'error', 'pool_address': addr, 'error': str(e)})

    processing_time = time.time() - start_time
    successful_results = [r for r in pool_results if r['status'] == 'success']
    
    print(f"\nProcess {rank}: Completed processing {len(successful_results)}/{len(pool_addresses)} pools in {processing_time:.2f} seconds.")
    print(f"Process {rank}: Now writing to HDF5 file...")

    # Atomic write phase: each process writes sequentially
    if MPI_AVAILABLE:
        for write_rank in range(size):
            if rank == write_rank:
                write_start = time.time()
                successful_writes = write_processed_pools_to_hdf5(pool_results, args.output_hdf5)
                write_time = time.time() - write_start
                print(f"Process {rank}: Successfully wrote {successful_writes} pools in {write_time:.2f} seconds.")
            # Barrier to ensure only one process writes at a time
            comm.Barrier()
    else:
        # Single process mode
        successful_writes = write_processed_pools_to_hdf5(pool_results, args.output_hdf5)
        print(f"Successfully wrote {successful_writes} pools.")

    elapsed_time = time.time() - start_time
    print(f"\nProcess {rank}: Total time {elapsed_time:.2f} seconds.")
    
    if rank == 0:
        print(f"Output saved to: {args.output_hdf5}")
        print("All processes completed successfully with atomic writes.")

if __name__ == "__main__":
    main()
