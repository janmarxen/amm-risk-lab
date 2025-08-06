# Data Download and Transformation Pipeline

This document explains the updated two-stage pipeline for downloading and transforming Uniswap pool data.

## Stage 1: Data Download (Raw Data)

The `run_data_download.py` script now saves raw data without feature engineering:

```bash
# Single process download
python python/ml/PLV/scripts/run_data_download.py \
    --api_key $API_KEY \
    --subgraph_id $SUBGRAPH_ID \
    --start_date "2023-01-01" \
    --end_date "2025-07-01" \
    --n_pools 1000
```

Or use the existing batch script:
```bash
bash python/ml/PLV/scripts/batch_run_data_download.sh
```

## Stage 2: Feature Engineering Transformation (Distributed)

The `run_data_transformation.py` script applies feature engineering with distributed processing:

### Local/Single Process:
```bash
python python/ml/PLV/scripts/run_data_transformation.py \
    --input_hdf5 /p/scratch/training2529/uniswap_pools_data.h5 \
    --output_hdf5 /p/scratch/training2529/uniswap_pools_data_transformed.h5 \
    --max_workers 16
```

### Distributed via SLURM (Recommended):
```bash
sbatch python/ml/PLV/scripts/batch_run_data_transform.sbatch
```

Or with custom parameters:
```bash
sbatch python/ml/PLV/scripts/batch_run_data_transform.sbatch \
    /path/to/input.h5 \
    /path/to/output.h5 \
    64
```

## How Distribution Works

With the SLURM configuration (8 nodes, 8 tasks):
- **Task 0**: Processes pools 0-124 (125 pools)
- **Task 1**: Processes pools 125-249 (125 pools)
- **Task 2**: Processes pools 250-374 (125 pools)
- ... and so on

Each task uses multithreading (64 threads per task) to process its assigned pools in parallel.

## Resource Allocation

- **8 nodes** × **1 task per node** = **8 parallel processes**
- **128 CPUs per task** with **64 threads per task** = efficient CPU utilization
- Each process handles ~125 pools (for 1000 total pools)
- All processes write to the same output HDF5 file with thread-safe locking

## Output

The transformed HDF5 file contains:
- All engineered features (60+ features per pool)
- Metadata with processing statistics per process
- Same structure as before, but with distributed processing history
