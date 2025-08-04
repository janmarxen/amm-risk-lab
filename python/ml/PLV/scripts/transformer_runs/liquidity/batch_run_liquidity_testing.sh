# Activate environment
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh

# Set the PYTHONPATH to include the project directory
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH
set -e

### Configuration ###
POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
MODEL_NAME="transformer_finetuned_liquidity_1_${POOL_ADDRESS}"

TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"


# Run testing script
python python/ml/PLV/scripts/run_testing.py \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --pool_address $POOL_ADDRESS \
    --model_name $MODEL_NAME
