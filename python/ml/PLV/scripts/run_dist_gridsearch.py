import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from random import shuffle
import random
import sys
from python.ml.PLV.data_io import LPsDataset, get_saved_pool_addresses
from python.ml.PLV.model import ZeroInflatedLSTM, ZeroInflatedTransformer
import itertools
import json

def main():

    args = parse_args()
    torch.manual_seed(args.seed)

    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    model_dir = args.result_dir if args.result_dir is not None else os.path.join("/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print("[run_gridsearch.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.target is not None:
        target = args.target
    else:
        print("[run_gridsearch.py] ERROR: --target argument must be specified.")
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
    if args.main_pool_address not in pool_addresses:
        pool_addresses.append(args.main_pool_address)

    n_lags_grid = [int(x) for x in args.n_lags_list.split(',')]
    batch_size_grid = [int(x) for x in args.batch_size_list.split(',')]
    dense_units_grid = [int(x) for x in args.dense_units_list.split(',')]
    lr_grid = [float(x) for x in args.lr_list.split(',')]
    epochs_grid = [int(x) for x in args.epochs_list.split(',')]

    if args.model_type == "lstm":
        model_class = ZeroInflatedLSTM
        param_grid = list(itertools.product(n_lags_grid, batch_size_grid, [int(x) for x in args.lstm_units_list.split(',')] if args.lstm_units_list else [32], dense_units_grid, lr_grid, epochs_grid))
        param_names = ["n_lags", "batch_size", "lstm_units", "dense_units", "lr", "epochs"]
    elif args.model_type == "transformer":
        model_class = ZeroInflatedTransformer
        d_model_grid = [int(x) for x in args.d_model_list.split(',')] if args.d_model_list else [32]
        num_heads_grid = [int(x) for x in args.num_heads_list.split(',')] if args.num_heads_list else [2]
        num_layers_grid = [int(x) for x in args.num_layers_list.split(',')] if args.num_layers_list else [2]
        dropout_grid = [float(x) for x in args.dropout_list.split(',')] if args.dropout_list else [0.1]
        param_grid = list(itertools.product(n_lags_grid, batch_size_grid, d_model_grid, num_heads_grid, num_layers_grid, dense_units_grid, dropout_grid, lr_grid, epochs_grid))
        param_names = ["n_lags", "batch_size", "d_model", "num_heads", "num_layers", "dense_units", "dropout", "lr", "epochs"]
    else:
        print(f"Unknown model_type: {args.model_type}")
        sys.exit(1)

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
        model_kwargs = dict(input_size=len(features)+1, n_lags=n_lags, dense_units=dense_units)
        if args.model_type == "lstm":
            model_kwargs["lstm_units"] = param_dict["lstm_units"]
        else:
            model_kwargs["d_model"] = param_dict["d_model"]
            model_kwargs["num_heads"] = param_dict["num_heads"]
            model_kwargs["num_layers"] = param_dict["num_layers"]
            model_kwargs["dropout"] = param_dict["dropout"]
        train_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=pool_addresses,
            features=features,
            target=target,
            n_lags=n_lags,
            split='train',
            split_dates=split_dates
        )
        val_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=pool_addresses,
            features=features,
            target=target,
            n_lags=n_lags,
            split='val',
            split_dates=split_dates
        )
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)), pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        model = model_class(**model_kwargs)
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
        # Finetuning on main pool
        finetune_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=[args.main_pool_address],
            features=features,
            target=target,
            n_lags=n_lags,
            split='train',
            split_dates=split_dates
        )
        finetune_val_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=[args.main_pool_address],
            features=features,
            target=target,
            n_lags=n_lags,
            split='val',
            split_dates=split_dates
        )
        if len(finetune_dataset) == 0 or len(finetune_val_dataset) == 0:
            print("Skipping finetuning for this grid point: no data for main pool.")
            continue
        finetune_train_loader = DataLoader(
            finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)), pin_memory=True, drop_last=True)
        finetune_val_loader = DataLoader(
            finetune_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        model.fit(
            train_loader=finetune_train_loader,
            epochs=epochs,
            lr=lr,
            verbose=1,
            val_loader=finetune_val_loader,
            early_stopping_patience=10
        )
        print("Finetuning complete.")
        finetune_test_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=[args.main_pool_address],
            features=features,
            target=target,
            n_lags=n_lags,
            split='test',
            split_dates=split_dates
        )
        finetune_test_loader = DataLoader(
            finetune_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        test_loss = model.evaluate(finetune_test_loader)
        print(f"Custom loss on main pool testing: {test_loss:.8f}")
        result = {
            'model_type': args.model_type,
            'params': param_dict,
            'test_loss': test_loss
        }
        # Save result to file
        out_path = os.path.join(model_dir, f"{args.model_name}_gpu{local_rank}_gridsearch_result_{i}.json")
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Grid search result written to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lags_list', type=str, required=True)
    parser.add_argument('--batch_size_list', type=str, required=True)
    parser.add_argument('--lstm_units_list', type=str, required=False)
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
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--main_pool_address', type=str, required=True)
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--result_dir', type=str, required=False, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    main()
