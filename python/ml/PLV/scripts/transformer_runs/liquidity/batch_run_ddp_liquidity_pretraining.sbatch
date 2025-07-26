#!/bin/bash
#SBATCH --job-name=liquidity_pretraining
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=00:30:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

## SBATCH --reservation=training2529 

# Get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Extracts the first hostname from the list of allocated nodes to use as the master address.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Modifies the master address to allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

# We activate our environemnt
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh

# Set the PYTHONPATH to include the project directory
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH
set -e

### Configuration ###
API_KEY="d1762c97d76a973e078c5536742bd237"
SUBGRAPH_ID="5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
N_LAGS=7
BATCH_SIZE=128
D_MODEL=32
NUM_HEADS=2
NUM_LAYERS=2
DENSE_UNITS=32
DROPOUT=0.1
LR=0.001
EPOCHS=50
N_POOLS=10000
MODEL_NAME="transformer_pretrained_liquidity"
FEATURES="price_return,price_volatility_3h,price_volatility_6h,price_volatility_24h,\
liquidity_volatility_3h,liquidity_volatility_6h,liquidity_volatility_24h,\
price_ma_3h,price_ma_6h,price_ma_24h,\
liquidity_ma_3h,liquidity_ma_6h,liquidity_ma_24h,\
hour,day_of_week,month,season"
TARGET="liquidity_return"
START_DATE="2023-01-01"
END_DATE="2025-07-01"
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"

# Launch a distributed training job across multiple nodes and GPUs
srun --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend c10d \
    --nproc_per_node=gpu \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    python/ml/PLV/scripts/run_ddp_pretraining.py \
    --n_lags $N_LAGS \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --dense_units $DENSE_UNITS \
    --dropout $DROPOUT \
    --lr $LR \
    --epochs $EPOCHS \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --model_name $MODEL_NAME \
    --n_pools $N_POOLS \
    --features "$FEATURES" \
    --target "$TARGET" \
    --model_type transformer"