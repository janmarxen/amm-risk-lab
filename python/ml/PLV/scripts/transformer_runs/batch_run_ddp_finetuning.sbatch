#!/bin/bash
#SBATCH --job-name=transformer_finetuning
#SBATCH --nodes=1
#SBATCH --gres=gpu:2  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=training2529
#SBATCH --partition=dc-gpu
#SBATCH --time=00:15:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

## SBATCH --reservation=training2529 

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

# Set the PYTHONPATH to include the project directory
export PYTHONPATH=/p/project1/training2529/marxen1/amm-risk-lab:$PYTHONPATH
set -e

### Configuration ###
HDF5_PATH="/p/scratch/training2529/uniswap_pools_data_transformed.h5"
API_KEY="d1762c97d76a973e078c5536742bd237"
SUBGRAPH_ID="5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

FINETUNE_LR=0.0005
FINETUNE_BATCH_SIZE=64

USE_VALIDATION=false
TRAIN_START="2023-01-01"
TRAIN_END="2025-06-14"
FINETUNE_EPOCHS=10

# USE_VALIDATION=true
# TRAIN_START="2023-01-01"
# TRAIN_END="2025-05-01"
# VAL_START="2025-05-02"
# VAL_END="2025-06-01"
# FINETUNE_EPOCHS=50

FINETUNE_POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"
PRETRAINED_MODEL_NAME="transformer_pretrained"
FINETUNED_MODEL_NAME="transformer_finetuned_${FINETUNE_POOL_ADDRESS}"

echo "=== Configuration ==="
echo "Pool: $FINETUNE_POOL_ADDRESS"
echo "Epochs: $FINETUNE_EPOCHS"
echo "Learning Rate: $FINETUNE_LR"
echo "Batch Size: $FINETUNE_BATCH_SIZE"
echo "Use Validation: $USE_VALIDATION"
echo "Training Period: $TRAIN_START to $TRAIN_END"
if [ "$USE_VALIDATION" = "true" ]; then
    echo "Validation Period: $VAL_START to $VAL_END"
fi
echo "========================"

# Launch distributed finetuning job
srun --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend c10d \
    --nproc_per_node=gpu \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    python/ml/PLV/scripts/run_ddp_finetuning.py \
    --hdf5_path $HDF5_PATH \
    --finetune_epochs $FINETUNE_EPOCHS \
    --finetune_lr $FINETUNE_LR \
    --finetune_batch_size $FINETUNE_BATCH_SIZE \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    $(if [ "$USE_VALIDATION" = "true" ]; then echo "--val_start $VAL_START --val_end $VAL_END"; fi) \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --finetune_pool_address $FINETUNE_POOL_ADDRESS \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --finetuned_model_name $FINETUNED_MODEL_NAME \
    $(if [ "$USE_VALIDATION" = "true" ]; then echo "--use_validation"; fi)"
