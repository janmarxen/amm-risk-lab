#!/bin/bash
#SBATCH --nodes=12
#SBATCH --gres=gpu:4  
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=04:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Extracts the first hostname from the list of allocated nodes to use as the master address.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

# Activate environment
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH
set -e

### Configuration ###
N_POOLS=200
MODEL_NAME="transformer_model_ddp_gs"
FEATURES="price_return,price_volatility_3h,price_volatility_6h,price_volatility_24h,liquidity_volatility_3h,liquidity_volatility_6h,liquidity_volatility_24h,price_ma_3h,price_ma_6h,price_ma_24h,liquidity_ma_3h,liquidity_ma_6h,liquidity_ma_24h,hour,day_of_week,month,season"
TARGET="liquidity_return"
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"
MAIN_POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"

# Grid search parameters (comma-separated lists)
N_LAGS_LIST="7,14"
BATCH_SIZE_LIST="32,64"
D_MODEL_LIST="32,64"
NUM_HEADS_LIST="2,4"
NUM_LAYERS_LIST="2,4"
DENSE_UNITS_LIST="16,32"
DROPOUT_LIST="0.1,0.2"
LR_LIST="0.001,0.0005"
EPOCHS_LIST="50,100"

# Launch distributed grid search with torchrun
srun --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    python/ml/PLV/scripts/run_ddp_gridsearch.py \
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
    --main_pool_address $MAIN_POOL_ADDRESS \
    --n_pools $N_POOLS \
    --features \"$FEATURES\" \
    --target \"$TARGET\" \
    --model_type transformer"
