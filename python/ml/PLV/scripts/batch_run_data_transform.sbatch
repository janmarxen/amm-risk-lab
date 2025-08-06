#!/bin/bash
#SBATCH --job-name=data_transform
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=128
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Load environment
echo "Loading environment..."
source envs/jureca0/activate.sh

# Set up parameters
INPUT_HDF5="${1:-/p/scratch/training2529/uniswap_pools_data_2.h5}"
OUTPUT_HDF5="${2:-/p/scratch/training2529/uniswap_pools_data_transformed.h5}"
MAX_WORKERS="${3:-128}"  # Threads per process

echo "Starting distributed data transformation job..."
echo "Input file: $INPUT_HDF5"
echo "Output file: $OUTPUT_HDF5"
echo "Max workers per task: $MAX_WORKERS"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Run the data transformation script with SLURM task distribution
srun --output=transform_task_%t.out --error=transform_task_%t.err \
    python3 -m python.ml.PLV.scripts.run_data_transformation \
    --input_hdf5 "$INPUT_HDF5" \
    --output_hdf5 "$OUTPUT_HDF5" \
    --max_workers "$MAX_WORKERS"

echo "Distributed data transformation job completed."
