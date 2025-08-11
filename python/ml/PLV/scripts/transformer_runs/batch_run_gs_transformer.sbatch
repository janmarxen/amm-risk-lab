#!/bin/bash
#SBATCH --job-name=gridsearch_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4        
#SBATCH --gres=gpu:4              
#SBATCH --cpus-per-task=8           
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=03:00:00             
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
N_POOLS=500  # Reduced for testing
MODEL_NAME="transformer_model_gs"
# Simplified feature set for testing
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

# Smaller time range for testing
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"

N_LAGS_LIST="5,8,10"                        # 2 values
BATCH_SIZE_LIST="1024,2048"                      # 1 value  
D_MODEL_LIST="16,32"                      # 2 values
NUM_HEADS_LIST="2,4"                      # 2 values
NUM_LAYERS_LIST="2,3"                       # 1 value
DENSE_UNITS_LIST="16,32,64"                     # 1 value
DROPOUT_LIST="0.3,0.4"                        # 1 value
LR_LIST="0.001"                           # 1 value
EPOCHS_LIST="50"                           # 1 epoch for testing

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