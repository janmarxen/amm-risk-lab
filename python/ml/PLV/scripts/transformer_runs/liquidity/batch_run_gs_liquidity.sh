#!/bin/bash
# batch_run_gs_transformer.sh
# This script automates grid search for the transformer pipeline.

set -e

### Configuration ###
API_KEY="d1762c97d76a973e078c5536742bd237"
SUBGRAPH_ID="5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
N_POOLS=200
MODEL_NAME="transformer_model_gs"
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

# Step 2: Grid search training
python3 -m python.ml.PLV.scripts.run_gridsearch \
    --n_lags_list "$N_LAGS_LIST" \
    --batch_size_list "$BATCH_SIZE_LIST" \
    --d_model_list "$D_MODEL_LIST" \
    --num_heads_list "$NUM_HEADS_LIST" \
    --num_layers_list "$NUM_LAYERS_LIST" \
    --dense_units_list "$DENSE_UNITS_LIST" \
    --dropout_list "$DROPOUT_LIST" \
    --lr_list "$LR_LIST" \
    --epochs_list "$EPOCHS_LIST" \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --model_name $MODEL_NAME \
    --main_pool_address $MAIN_POOL_ADDRESS \
    --n_pools $N_POOLS \
    --features "$FEATURES" \
    --target "$TARGET" \
    --model_type transformer
