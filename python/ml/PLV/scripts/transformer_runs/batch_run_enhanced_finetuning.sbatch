#!/bin/bash
#SBATCH --job-name=enhanced_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=00:15:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Activate environment
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH

export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0

echo "=== Enhanced Finetuning Job Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

### Configuration ###
HDF5_PATH="/p/scratch/training2529/uniswap_pools_data_transformed.h5"

# Enhanced finetuning parameters - designed for better adaptation
FINETUNE_EPOCHS=60          # More epochs for gradual adaptation  
FINETUNE_LR=0.00005         # Very low learning rate (50% lower than before)
FINETUNE_BATCH_SIZE=32      # Smaller batch size for more stable gradients

START_DATE="2023-01-01"
END_DATE="2025-07-01"
TRAIN_START="2023-01-01"
TRAIN_END="2025-05-01"
VAL_START="2025-05-02"
VAL_END="2025-06-01"
TEST_START="2025-06-02"
TEST_END="2025-07-01"

FINETUNE_POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
PRETRAINED_MODEL="transformer_pretrained"
FINETUNED_MODEL="transformer_finetuned_${FINETUNE_POOL_ADDRESS}"

echo "=== Configuration ==="
echo "Pool: $FINETUNE_POOL_ADDRESS"
echo "Epochs: $FINETUNE_EPOCHS"
echo "Learning Rate: $FINETUNE_LR"
echo "Batch Size: $FINETUNE_BATCH_SIZE"
echo "Training Period: $TRAIN_START to $TRAIN_END"
echo "Validation Period: $VAL_START to $VAL_END"

python python/ml/PLV/scripts/run_enhanced_finetuning.py \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --val_start $VAL_START \
    --val_end $VAL_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --finetune_pool_address $FINETUNE_POOL_ADDRESS \
    --finetune_epochs $FINETUNE_EPOCHS \
    --finetune_lr $FINETUNE_LR \
    --finetune_batch_size $FINETUNE_BATCH_SIZE \
    --pretrained_model_name $PRETRAINED_MODEL \
    --finetuned_model_name $FINETUNED_MODEL \
    --hdf5_path $HDF5_PATH \
    --seed 42

echo "=== Enhanced Finetuning Complete ==="
