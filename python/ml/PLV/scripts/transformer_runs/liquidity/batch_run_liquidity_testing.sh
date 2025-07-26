# Activate environment
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh

# Set the PYTHONPATH to include the project directory
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH
set -e

### Configuration ###
API_KEY="d1762c97d76a973e078c5536742bd237"
SUBGRAPH_ID="5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
N_LAGS=7
D_MODEL=32
NUM_HEADS=2
NUM_LAYERS=2
DENSE_UNITS=32
DROPOUT=0.1
FEATURES="price_return,price_volatility_3h,price_volatility_6h,price_volatility_24h,liquidity_volatility_3h,liquidity_volatility_6h,liquidity_volatility_24h,price_ma_3h,price_ma_6h,price_ma_24h,liquidity_ma_3h,liquidity_ma_6h,liquidity_ma_24h,hour,day_of_week,month,season"
TARGET="liquidity_return"
START_DATE="2023-01-01"
END_DATE="2025-07-01"
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"
MODEL_NAME="transformer_finetuned_liquidity_${POOL_ADDRESS}"

# Run testing script
python python/ml/PLV/scripts/run_testing.py \
    --n_lags $N_LAGS \
    --d_model $D_MODEL \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --dense_units $DENSE_UNITS \
    --dropout $DROPOUT \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --pool_address $POOL_ADDRESS \
    --features "$FEATURES" \
    --target "$TARGET" \
    --model_type transformer \
    --model_name $MODEL_NAME 
