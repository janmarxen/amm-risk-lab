#!/bin/bash
#SBATCH --job-name=data_transform_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=128
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=00:30:00
#SBATCH --output=data_transform_mpi_%j.out
#SBATCH --error=data_transform_mpi_%j.err

# Load environment
echo "Loading environment..."
source envs/jureca0/activate.sh

# Set up parameters
INPUT_HDF5="${1:-/p/scratch/training2529/uniswap_pools_data.h5}"
OUTPUT_HDF5="${2:-/p/scratch/training2529/uniswap_pools_data_transformed.h5}"
MAX_WORKERS="${3:-128}"

echo "Starting MPI-based data transformation job..."
echo "Input file: $INPUT_HDF5"
echo "Output file: $OUTPUT_HDF5"
echo "Max workers per process: $MAX_WORKERS"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total MPI processes: $SLURM_NTASKS"
echo "CPUs per process: $SLURM_CPUS_PER_TASK"

# Run with mpiexec and per-process output files
echo "Starting MPI execution with mpiexec..."
mpiexec -n $SLURM_NTASKS \
    --output-filename transform_rank_%r_job_${SLURM_JOB_ID} \
    python3 -m python.ml.PLV.scripts.run_data_transformation \
    --input_hdf5 "$INPUT_HDF5" \
    --output_hdf5 "$OUTPUT_HDF5" \
    --max_workers "$MAX_WORKERS"

echo "MPI-based data transformation job completed."
