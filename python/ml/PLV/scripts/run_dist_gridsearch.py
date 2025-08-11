"""
run_dist_gridsearch.py

Distributed grid search script for hyperparameter optimization of Uniswap V3 ML models.

High-level steps:
1. Parse command-line arguments for grid search parameters and data configuration.
2. Generate all hyperparameter combinations and distribute across available GPUs.
3. For each hyperparameter combination:
   a. Pretrain model on multiple pools using per-pool scaling (like pretraining script)
   b. Evaluate model on pretraining validation data to get grid search loss
4. Save grid search results for each combination to individual JSON files.
5. Results can be collected later to find the best hyperparameter combination.

Note: This script uses the pretraining validation loss as the optimization metric, which is
efficient and provides good hyperparameter selection across diverse pool data without
requiring individual pool-specific finetuning for each grid search combination.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from random import shuffle
import random
import sys
from python.ml.PLV.data_io import LPsDataset, get_saved_pool_addresses
from python.ml.PLV.model import ZeroInflatedTransformer
from python.utils.distributed_utils import save_gridsearch_result
import itertools

def main(args):

    hdf5_path = args.hdf5_path
    model_dir = args.result_dir if args.result_dir is not None else os.path.join("/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print("[run_gridsearch.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.targets is not None:
        targets = [t.strip() for t in args.targets.split(',')]
        if len(targets) != 2:
            print("[run_gridsearch.py] ERROR: Exactly 2 targets must be specified for multi-task model.")
            sys.exit(1)
    else:
        print("[run_gridsearch.py] ERROR: --targets argument must be specified with exactly 2 targets.")
        sys.exit(1)
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }
    pool_addresses = get_saved_pool_addresses(hdf5_path)
    print(f"Number of pools in HDF5: {len(pool_addresses)}")
    random.seed(42)
    shuffle(pool_addresses)
    N = args.n_pools
    pool_addresses = pool_addresses[:N]

    n_lags_grid = [int(x) for x in args.n_lags_list.split(',')]
    batch_size_grid = [int(x) for x in args.batch_size_list.split(',')]
    dense_units_grid = [int(x) for x in args.dense_units_list.split(',')]
    lr_grid = [float(x) for x in args.lr_list.split(',')]
    epochs_grid = [int(x) for x in args.epochs_list.split(',')]
    d_model_grid = [int(x) for x in args.d_model_list.split(',')] if args.d_model_list else [32]
    num_heads_grid = [int(x) for x in args.num_heads_list.split(',')] if args.num_heads_list else [2]
    num_layers_grid = [int(x) for x in args.num_layers_list.split(',')] if args.num_layers_list else [2]
    dropout_grid = [float(x) for x in args.dropout_list.split(',')] if args.dropout_list else [0.1]
    param_grid = list(itertools.product(n_lags_grid, batch_size_grid, d_model_grid, num_heads_grid, num_layers_grid, dense_units_grid, dropout_grid, lr_grid, epochs_grid))
    param_names = ["n_lags", "batch_size", "d_model", "num_heads", "num_layers", "dense_units", "dropout", "lr", "epochs"]

    print(f"Grid search over {len(param_grid)} combinations.")
    # Assign grid points to processes by GPU id
    local_rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    print(f"[run_gridsearch.py] Local rank: {local_rank}, World size: {world_size}")
    for i, params in enumerate(param_grid):
        if i % world_size != local_rank:
            continue
        param_dict = dict(zip(param_names, params))
        print(f"\n[GridSearch] Combination {i+1}/{len(param_grid)}: " + ", ".join(f"{k}={v}" for k, v in param_dict.items()))
        n_lags = param_dict["n_lags"]
        batch_size = param_dict["batch_size"]
        dense_units = param_dict["dense_units"]
        lr = param_dict["lr"]
        epochs = param_dict["epochs"]
        model_kwargs = dict(input_size=len(features)+len(targets), n_lags=n_lags, dense_units=dense_units)
        model_kwargs["d_model"] = param_dict["d_model"]
        model_kwargs["num_heads"] = param_dict["num_heads"]
        model_kwargs["num_layers"] = param_dict["num_layers"]
        model_kwargs["dropout"] = param_dict["dropout"]

        # Pretraining with per-pool scaling (like pretraining script)
        train_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=pool_addresses,
            features=features,
            targets=targets,
            n_lags=n_lags,
            split='train',
            split_dates=split_dates,
            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4))
        )
        val_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=pool_addresses,
            features=features,
            targets=targets,
            n_lags=n_lags,
            split='val',
            split_dates=split_dates,
            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4))
        )
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)), pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        model = ZeroInflatedTransformer(**model_kwargs)
        # Pretraining
        model.fit(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            verbose=1,
            val_loader=val_loader,
            early_stopping_patience=20
        )
        print('Model training complete.')
        
        # Evaluate on pretraining validation data for grid search results
        val_loss = model.evaluate(val_loader)/batch_size
        print(f"Normalized validation loss on pretraining pools: {val_loss:.8f}")
        result = {
            'combination_id': i,
            'gpu_rank': local_rank,
            'params': param_dict,
            'val_loss': val_loss
        }
        
        # Write to single results file
        results_file = os.path.join(model_dir, f"{args.model_name}_gridsearch_results.jsonl")
        save_gridsearch_result(result, results_file)
        print(f"Grid search result written to {results_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 file containing pool data')
    parser.add_argument('--n_lags_list', type=str, required=True)
    parser.add_argument('--batch_size_list', type=str, required=True)
    parser.add_argument('--d_model_list', type=str, required=False)
    parser.add_argument('--num_heads_list', type=str, required=False)
    parser.add_argument('--num_layers_list', type=str, required=False)
    parser.add_argument('--dense_units_list', type=str, required=True)
    parser.add_argument('--dropout_list', type=str, required=False)
    parser.add_argument('--lr_list', type=str, required=True)
    parser.add_argument('--epochs_list', type=str, required=True)
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=False, default="model_gs")
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--targets', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--result_dir', type=str, required=False, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    main(args)
