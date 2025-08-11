#!/bin/bash
#SBATCH --job-name=gridsearch_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        
#SBATCH --gres=gpu:4              
#SBATCH --cpus-per-task=8           
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=00:20:00             
#SBATCH --output=%j.out
#SBATCH --error=%j.err
##SBATCH --output=/dev/null
##SBATCH --error=/dev/null

LOG_DIR=/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models/gs_logs
mkdir -p "$LOG_DIR"

# Activate environment BEFORE redirecting output
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH

# Now redirect output after environment is set up
exec > "$LOG_DIR/job_${SLURM_JOB_ID}_task_${SLURM_PROCID}.out" 2> "$LOG_DIR/job_${SLURM_JOB_ID}_task_${SLURM_PROCID}.err"

echo "=== Grid Search Job Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_PROCID"
echo "Time: $(date)"

export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0

# Remove set -e to prevent immediate exit on errors
# set -e

# Create output directories
mkdir -p /p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models/gs_logs
# Note: gs_results directory no longer needed - using single file in models/

### Configuration ###
HDF5_PATH="/p/scratch/training2529/uniswap_pools_data_transformed.h5"
N_POOLS=5  # Reduced for testing
MODEL_NAME="transformer_model_gs"
# Simplified feature set for testing
FEATURES="price_return,liquidity_return,volume_return,price_volatility_3h,liquidity_volatility_3h,volume_volatility_3h"
TARGETS="liquidity_return,volume_return"

# Smaller time range for testing
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"

# Simplified grid for testing (Total combinations: 2×1×2×2×1×1×1 = 8)
N_LAGS_LIST="3,5"                        # 2 values
BATCH_SIZE_LIST="32"                      # 1 value  
D_MODEL_LIST="32,64"                      # 2 values
NUM_HEADS_LIST="2,4"                      # 2 values
NUM_LAYERS_LIST="2"                       # 1 value
DENSE_UNITS_LIST="32"                     # 1 value
DROPOUT_LIST="0.1"                        # 1 value
LR_LIST="0.001"                           # 1 value
EPOCHS_LIST="1"                           # 1 epoch for testing

# Launch one Python process per task (per GPU)
srun --cpu_bind=none \
    --output=$LOG_DIR/job_${SLURM_JOB_ID}_task_%t.out \
     --error=$LOG_DIR/job_${SLURM_JOB_ID}_task_%t.err \
    python python/ml/PLV/scripts/run_dist_gridsearch.py \
    --hdf5_path $HDF5_PATH \
    --n_lags_list $N_LAGS_LIST \
    --batch_size_list $BATCH_SIZE_LIST \
    --d_model_list $D_MODEL_LIST \
    --num_heads_list $NUM_HEADS_LIST \
    --num_layers_list $NUM_LAYERS_LIST \
    --dense_units_list $DENSE_UNITS_LIST \
    --dropout_list $DROPOUT_LIST \
    --lr_list $LR_LIST \
    --epochs_list $EPOCHS_LIST \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --model_name $MODEL_NAME \
    --n_pools $N_POOLS \
    --features "$FEATURES" \
    --targets "$TARGETS"