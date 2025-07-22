import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from random import shuffle
import random
import sys
from python.ml.PLV.data_io import LPsDataset, get_saved_pool_addresses
from python.ml.PLV.model import ZeroInflatedLSTM, ZeroInflatedTransformer
from python.utils.distributed_utils import *
import itertools

def main_worker(rank, world_size, args):
    local_rank, rank, device = setup()

    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    model_dir = os.path.join("/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print0("[run_ddp_gridsearch.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.target is not None:
        target = args.target
    else:
        print0("[run_ddp_gridsearch.py] ERROR: --target argument must be specified.")
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
    print0(f"Number of pools in HDF5: {len(pool_addresses)}")
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
        print0(f"Unknown model_type: {args.model_type}")
        sys.exit(1)

    print0(f"Grid search over {len(param_grid)} combinations.")
    # Distribute grid points across ranks
    results = []
    for i, params in enumerate(param_grid):
        if i % dist.get_world_size() != dist.get_rank():
            continue  # Only process grid points assigned to this rank
        param_dict = dict(zip(param_names, params))
        print0(f"\n[GridSearch] Combination {i+1}/{len(param_grid)}: " + ", ".join(f"{k}={v}" for k, v in param_dict.items()))
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
        print0(f"Number of training samples: {len(train_dataset)}")
        print0(f"Number of validation samples: {len(val_dataset)}")
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, seed=args.seed if hasattr(args, 'seed') else 42)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)), pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, drop_last=False)
        model = model_class(**model_kwargs)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model.module.fit_distributed(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            verbose=1 if rank == 0 else 0,
            val_loader=val_loader,
            early_stopping_patience=20,
            device=device
        )
        print0('Training complete.')
        val_loss = model.module.evaluate_distributed(val_loader, device=device)
        result = {
            'params': param_dict,
            'val_loss': val_loss
        }
        results.append(result)

    # Gather all results to rank 0
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, results)
    if dist.get_rank() == 0:
        # Flatten and sort by grid index
        all_results = []
        for rlist in gathered_results:
            all_results.extend(rlist)
        all_results = sorted(all_results, key=lambda r: [r['params'][k] for k in param_names])
        import json
        out_path = os.path.join(model_dir, f"{args.model_name}_ddp_gridsearch_results.json")
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print0(f"All grid search results written to {out_path}")
    destroy_process_group()

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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    main_worker(0, world_size, args)
