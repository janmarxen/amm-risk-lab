
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


def main(args):
    local_rank, rank, device = setup()

    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    model_dir = os.path.join("/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models")
    model_path = os.path.join(model_dir, f"{args.model_name}.pt")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print0("[run_ddp_training.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.target is not None:
        target = args.target
    else:
        print0("[run_ddp_training.py] ERROR: --target argument must be specified.")
        sys.exit(1)
    n_lags = args.n_lags
    batch_size = args.batch_size
    lstm_units = args.lstm_units
    dense_units = args.dense_units
    lr = args.lr
    epochs = args.epochs
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }

    print0(f"[run_ddp_training.py] Configuration:")
    for k, v in vars(args).items():
        print0(f"  {k}: {v}")
    print0("Preparing dataset...")
    pool_addresses = get_saved_pool_addresses(hdf5_path)
    print0(f"Number of pools in HDF5: {len(pool_addresses)}")
    random.seed(42)
    shuffle(pool_addresses)
    N = args.n_pools
    pool_addresses = pool_addresses[:N]
    print0("Loading dataset...")
    train_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=pool_addresses,
        features=features,
        target=target,
        n_lags=n_lags,
        split='train',
        split_dates=split_dates,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)),
    )
    val_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=pool_addresses,
        features=features,
        target=target,
        n_lags=n_lags,
        split='val',
        split_dates=split_dates,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)),
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
    input_size = len(features) + 1
    if args.model_type == "lstm":
        model = ZeroInflatedLSTM(
            input_size=input_size,
            n_lags=n_lags,
            lstm_units=lstm_units,
            dense_units=dense_units
        )
    else:
        model = ZeroInflatedTransformer(
            input_size=input_size,
            n_lags=n_lags,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dense_units=dense_units,
            dropout=args.dropout
        )
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
    save0(model, model_path)
    destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lags', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lstm_units', type=int, required=False, help='Number of LSTM units (required for LSTM)')
    parser.add_argument('--dense_units', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=False, default="model")
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'], help='Model type to use')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=False, default=None, help='Comma-separated list of features')
    parser.add_argument('--target', type=str, required=False, default=None, help='Target column name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    main(args)
