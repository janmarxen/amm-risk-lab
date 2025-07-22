#!/bin/bash
# batch_run_transformer.sh
# This script automates training and testing for the transformer pipeline.

set -e

### Configuration ###
API_KEY="d1762c97d76a973e078c5536742bd237"
SUBGRAPH_ID="5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
N_POOLS=50000
START_DATE="2023-01-01"
END_DATE="2025-07-01"
MAIN_POOL_ADDRESS="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"


# Activate environment
source /p/project1/training2529/marxen1/amm-risk-lab/envs/jureca0/activate.sh

# Download data 
python3 -m python.ml.PLV.scripts.run_data_download \
    --api_key $API_KEY \
    --subgraph_id $SUBGRAPH_ID \
    --start_date $START_DATE \
    --end_date $END_DATE \
    --main_pool_address $MAIN_POOL_ADDRESS \
    --n_pools $N_POOLS

