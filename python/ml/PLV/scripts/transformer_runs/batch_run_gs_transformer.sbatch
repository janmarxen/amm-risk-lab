#!/bin/bash
#SBATCH --job-name=gridsearch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4         # 1 task per GPU
#SBATCH --gres=gpu:4               
#SBATCH --cpus-per-task=32
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
LOG_DIR=/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models/gs_logs
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/job_${SLURM_JOB_ID}_task_${SLURM_PROCID}.out" 2> "$LOG_DIR/job_${SLURM_JOB_ID}_task_${SLURM_PROCID}.err"

# Activate environment
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH

export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0

set -e

# Create output directories
mkdir -p /p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models/gs_logs
# Note: gs_results directory no longer needed - using single file in models/

### Configuration ###
N_POOLS=1000
MODEL_NAME="transformer_model_gs"
FEATURES="price_return,price_volatility_3h,price_volatility_6h,price_volatility_24h,\
liquidity_volatility_3h,liquidity_volatility_6h,liquidity_volatility_24h,\
volume_volatility_3h,volume_volatility_6h,volume_volatility_24h,\
price_ma_3h,price_ma_6h,price_ma_24h,\
liquidity_ma_3h,liquidity_ma_6h,liquidity_ma_24h,\
hour,day_of_week,month,season"
TARGETS="liquidity_return,volume_return"

TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"

# Grid search hyperparameters
N_LAGS_LIST="3,5,8"
BATCH_SIZE_LIST="1024"
D_MODEL_LIST="16"
NUM_HEADS_LIST="4"
NUM_LAYERS_LIST="3"
DENSE_UNITS_LIST="32"
DROPOUT_LIST="0.3"
LR_LIST="0.001"
EPOCHS_LIST="40"

# Launch one Python process per task (per GPU)
srun --cpu_bind=none \
    --output=$LOG_DIR/job_${SLURM_JOB_ID}_task_%t.out \
     --error=$LOG_DIR/job_${SLURM_JOB_ID}_task_%t.err \
    python python/ml/PLV/scripts/run_dist_gridsearch.py \
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
    --target "$TARGET"
    # Note: Removed --result_dir since we're using the default models directory
