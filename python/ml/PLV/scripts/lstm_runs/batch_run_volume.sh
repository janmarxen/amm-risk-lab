#!/bin/bash
# batch_run.sh
# This script automates the process of downloading data, training, and testing a model.

set -e

### Configuration ###
# Keys and IDs for the subgraph and API
API_KEY="d1762c97d76a973e078c5536742bd237"
SUBGRAPH_ID="5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
N_LAGS=7
BATCH_SIZE=64
LSTM_UNITS=16
DENSE_UNITS=32 
LR=0.001
EPOCHS=50
N_POOLS=200 # Number of pools to download and train on 
MODEL_NAME="lstm_model_volume_0" # It will be saved as <model name>.pt in the models directory
# Features and target
FEATURES="price_return,price_volatility_3h,price_volatility_6h,price_volatility_24h,\
volume_volatility_3h,volume_volatility_6h,volume_volatility_24h,\
price_ma_3h,price_ma_6h,price_ma_24h,\
volume_ma_3h,volume_ma_6h,volume_ma_24h,\
hour,day_of_week,month,season"
TARGET="volume_return"
# Date ranges for data download 
START_DATE="2023-01-01"
END_DATE="2025-07-01"
# Date ranges for training, validation, and testing
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"
# Main pool address for the model and finetuning parameters
MAIN_POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
FINETUNE_EPOCHS=200
FINETUNE_LR=0.001
FINETUNE_BATCH_SIZE=16

# # Step 1: Download data
# python3 -m python.ml.PLV.scripts.run_data_download \
#     --api_key $API_KEY \
#     --subgraph_id $SUBGRAPH_ID \
#     --start_date $START_DATE \
#     --end_date $END_DATE \
#     --main_pool_address $MAIN_POOL_ADDRESS \
#     --n_pools $N_POOLS


# Step 2: Train model
python3 -m python.ml.PLV.scripts.run_training \
    --n_lags $N_LAGS \
    --batch_size $BATCH_SIZE \
    --lstm_units $LSTM_UNITS \
    --dense_units $DENSE_UNITS \
    --lr $LR \
    --epochs $EPOCHS \
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
    --target "$TARGET"


# Step 3: Test model
python3 -m python.ml.PLV.scripts.run_testing \
    --n_lags $N_LAGS \
    --batch_size $BATCH_SIZE \
    --lstm_units $LSTM_UNITS \
    --dense_units $DENSE_UNITS \
    --api_key $API_KEY \
    --subgraph_id $SUBGRAPH_ID \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --model_name $MODEL_NAME \
    --main_pool_address $MAIN_POOL_ADDRESS \
    --finetune_epochs $FINETUNE_EPOCHS \
    --finetune_lr $FINETUNE_LR \
    --finetune_batch_size $FINETUNE_BATCH_SIZE \
    --features "$FEATURES" \
    --target "$TARGET"
