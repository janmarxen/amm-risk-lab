"""
run_ddp_pretraining.py

Distributed pretraining script for Uniswap V3 ML models using PyTorch DDP with dataset sharding.

High-level steps:
1. Initialize distributed process group and set up device for each rank.
2. Parse command-line arguments for model/data configuration.
3. Load pool addresses and shuffle/select a subset for training.
4. Shard pools across processes - each process gets a distinct subset of pools.
5. Create standard LPsDataset for each process using its assigned pools.
6. Build the Transformer model and wrap with DistributedDataParallel.
7. Train the model using distributed data loaders (no DistributedSampler needed).
8. Save the trained model (only on rank 0).
9. Clean up and destroy the process group.
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
from python.utils.distributed_utils import *


def main(args):
    local_rank, rank, device = setup()

    hdf5_path = args.input_hdf5
    model_dir = os.path.join("/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models")
    model_path = os.path.join(model_dir, f"{args.model_name}.pt")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print0("[run_ddp_pretraining.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.targets is not None:
        targets = [t.strip() for t in args.targets.split(',')]
    else:
        print0("[run_ddp_pretraining.py] ERROR: --targets argument must be specified.")
        sys.exit(1)
    
    # Always use multi-task architecture with exactly 2 targets
    if len(targets) != 2:
        print0("[run_ddp_pretraining.py] ERROR: Must specify exactly 2 targets for multi-task architecture.")
        sys.exit(1)
    n_lags = args.n_lags
    batch_size = args.batch_size
    dense_units = args.dense_units
    lr = args.lr
    epochs = args.epochs
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start if args.use_validation else None,
        'val_end': args.val_end if args.use_validation else None,
        'test_start': args.test_start,
        'test_end': args.test_end
    }

    print0(f"[run_ddp_pretraining.py] Configuration:")
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
    
    # Shard pools across processes for distributed training
    world_size = torch.distributed.get_world_size()
    pools_per_process = len(pool_addresses) // world_size
    start_idx = rank * pools_per_process
    
    if rank == world_size - 1:
        # Last process takes any remaining pools
        end_idx = len(pool_addresses)
    else:
        end_idx = start_idx + pools_per_process
    
    process_pool_addresses = pool_addresses[start_idx:end_idx]
    print(f"Process {rank}: Using pools {start_idx}-{end_idx-1} ({len(process_pool_addresses)} pools)")
    
    # Create datasets - standard dataset with per-process pool sharding
    train_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=process_pool_addresses,
        features=features,
        targets=targets,
        split_dates=split_dates,
        split='train',
        n_lags=n_lags,
        verbose=1 if rank == 0 else 0
    )
    
    val_dataset = None
    local_val_samples = 0
    if args.use_validation:
        val_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=process_pool_addresses,
            features=features,
            targets=targets,
            split_dates=split_dates,
            split='val',
            n_lags=n_lags,
            verbose=1 if rank == 0 else 0
        )
        local_val_samples = len(val_dataset)
    
    # Print dataset info - gather statistics from sharded datasets
    local_train_samples = len(train_dataset)
    
    # Use atomic_print to show per-process statistics
    from python.utils.distributed_utils import atomic_print
    atomic_print(f"Local samples - Train: {local_train_samples}, Val: {local_val_samples}, Pools: {len(process_pool_addresses)}")
    
    # Calculate total across all processes (only print on rank 0)
    total_train_samples = torch.tensor(local_train_samples, device=device)
    total_val_samples = torch.tensor(local_val_samples, device=device)
    torch.distributed.all_reduce(total_train_samples, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_val_samples, op=torch.distributed.ReduceOp.SUM)
    
    print0(f"Total training samples across all processes: {total_train_samples.item()}")
    if args.use_validation:
        print0(f"Total validation samples across all processes: {total_val_samples.item()}")
    else:
        print0("Validation disabled - training without early stopping")
    print0(f"Total number of pools across all processes: {len(pool_addresses)}")
    print0(f"Number of features: {len(features)}")
    print0(f"Using validation: {args.use_validation}")
    
    # Create DataLoaders - standard dataset with dataset-level sharding (no DistributedSampler)
    num_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 4))
    print0("Creating DataLoaders for standard Dataset with dataset-level sharding...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Each process shuffles its own shard
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if args.use_validation:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    # Calculate input size: features + target lags (2 targets)
    input_size = len(features) + len(targets)
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
    
    # Configure training parameters based on validation usage
    early_stopping_patience = 20 if args.use_validation else None
    
    model.module.fit_distributed(
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        verbose=1 if rank == 0 else 0,
        val_loader=val_loader,
        early_stopping_patience=early_stopping_patience,
        device=device
    )
    print0('Training complete.')
    # Save the full model taking DDP into account
    save_full_model(model, None, model_path)
    # Save transformer model hyperparameters to JSON for later finetuning
    save_model_arch0(
        model_path,
        n_lags=n_lags,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dense_units=dense_units,
        dropout=args.dropout,
        features=features,
        targets=targets
    )
    destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_hdf5', type=str, required=False, 
                       default="/p/scratch/training2529/uniswap_pools_data_transformed.h5",
                       help="Path to input HDF5 file with transformed pool data")
    parser.add_argument('--n_lags', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dense_units', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=False, help='Validation start date (required if --use_validation is set)')
    parser.add_argument('--val_end', type=str, required=False, help='Validation end date (required if --use_validation is set)')
    parser.add_argument('--test_start', type=str, required=False, help='Test start date (optional)')
    parser.add_argument('--test_end', type=str, required=False, help='Test end date (optional)')
    parser.add_argument('--model_name', type=str, required=False, default="model")
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=False, default=None, help='Comma-separated list of features')
    parser.add_argument('--targets', type=str, required=True, help='Comma-separated list of exactly 2 target column names')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_validation', action='store_true', help='Enable validation and early stopping during training')
    
    args = parser.parse_args()
    
    # Validate that validation dates are provided if validation is enabled
    if args.use_validation and (not args.val_start or not args.val_end):
        parser.error("--val_start and --val_end are required when --use_validation is set")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    main(args)
