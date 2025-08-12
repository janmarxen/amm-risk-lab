#!/bin/bash
#SBATCH --job-name=transformer_pretraining
#SBATCH --nodes=2
#SBATCH --gres=gpu:4 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=02:00:00
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

export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0

set -e

### Configuration ###
INPUT_HDF5="/p/scratch/training2529/uniswap_pools_data_transformed.h5"

MODEL_NAME="transformer_pretrained"
N_LAGS=5       
BATCH_SIZE=1024 
D_MODEL=32      
NUM_HEADS=4     
NUM_LAYERS=3    
DENSE_UNITS=16  
DROPOUT=0.3     
LR=0.001               

N_POOLS=10000
FEATURES="price_return,\
price_volatility_3h,price_volatility_6h,price_volatility_24h,\
liquidity_volatility_3h,liquidity_volatility_6h,liquidity_volatility_24h,\
volume_volatility_3h,volume_volatility_6h,volume_volatility_24h,\
price_ma_3h,price_ma_6h,price_ma_24h,\
liquidity_ma_3h,liquidity_ma_6h,liquidity_ma_24h,\
volume_ma_3h,volume_ma_6h,volume_ma_24h,\
price_volume_corr_6h,price_volume_corr_24h,price_liquidity_corr_6h,\
volume_liquidity_return_ratio,volume_liquidity_ratio_ma_6h,volume_liquidity_ratio_volatility,\
price_volatility_to_volume_ratio,liquidity_return_depth_proxy,market_impact_proxy,price_efficiency,\
price_momentum_3h,price_momentum_12h,price_momentum_24h,\
liquidity_momentum_6h,liquidity_momentum_24h,\
volume_momentum_6h,volume_momentum_24h,\
price_deviation_from_ma_6h,price_deviation_from_ma_24h,liquidity_deviation_from_ma_6h,\
high_vol_regime_3h,high_vol_regime_24h,vol_cluster_indicator,vol_term_structure,\
liquidity_return_concentration,liquidity_return_percentile_6h,liquidity_return_percentile_24h,\
liquidity_stress,liquidity_abundance,liquidity_stability,\
hour,day_of_week,month,season,\
hour_sin,hour_cos,day_sin,day_cos,\
us_market_hours,asia_market_hours,weekend,\
price_rsi_6h,price_rsi_24h,liquidity_rsi_6h,volume_rsi_6h,\
price_bb_upper_6h,price_bb_lower_6h,price_bb_position,\
price_shock_3h,price_shock_6h,volume_spike,volume_drought,\
liquidity_drain,liquidity_injection"
TARGETS="liquidity_return,volume_return"


USE_VALIDATION=false
TRAIN_START="2023-01-01"
TRAIN_END="2025-06-14"
EPOCHS=97 

# USE_VALIDATION=true
# TRAIN_START="2023-01-01"
# TRAIN_END="2025-05-01"
# VAL_START="2025-05-02"
# VAL_END="2025-06-01"
# EPOCHS=200 

echo "=== Pretraining Configuration ==="
echo "Model: $MODEL_NAME"
echo "N_LAGS: $N_LAGS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "Use Validation: $USE_VALIDATION"
echo "N_POOLS: $N_POOLS"
echo "Training Period: $TRAIN_START to $TRAIN_END"
if [ "$USE_VALIDATION" = "true" ]; then
    echo "Validation Period: $VAL_START to $VAL_END"
fi
echo "========================"

# Launch a distributed training job across multiple nodes and GPUs
srun --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend c10d \
    --nproc_per_node=gpu \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    python/ml/PLV/scripts/run_ddp_pretraining.py \
    --input_hdf5 $INPUT_HDF5 \
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
    $(if [ "$USE_VALIDATION" = "true" ]; then echo "--val_start $VAL_START --val_end $VAL_END"; fi) \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --model_name $MODEL_NAME \
    --n_pools $N_POOLS \
    --features "$FEATURES" \
    --targets "$TARGETS" \
    $(if [ "$USE_VALIDATION" = "true" ]; then echo "--use_validation"; fi)
"