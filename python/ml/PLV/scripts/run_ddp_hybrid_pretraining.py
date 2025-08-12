#!/usr/bin/env python3
"""
Distributed pretraining script for Hybrid Multi-task Learning:
- Task 1 (liquidity_return): Zero-inflated (classification + regression)
- Task 2 (volume_return): Standard regression only

Uses PyTorch DDP for distributed training across multiple GPUs/nodes.
"""

import os
import sys
import argparse
import time
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append('/p/project1/training2529/marxen1/amm-risk-lab')
from python.ml.PLV.data_io import fit_scalers, save_scalers, HybridLPsDataset
from python.ml.PLV.model import HybridTransformer

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    return rank, world_size, device

def save_hybrid_model(model, model_path, arch_config):
    """Save hybrid model with architecture configuration."""
    # Save model state dict
    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), model_path)
    
    # Save architecture config
    arch_path = os.path.splitext(model_path)[0] + '_arch.json'
    with open(arch_path, 'w') as f:
        json.dump(arch_config, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Architecture saved to {arch_path}")

def main():
    parser = argparse.ArgumentParser(description='Distributed Hybrid Pretraining')
    
    # Data arguments
    parser.add_argument('--input_hdf5', type=str, required=True, help='Path to input HDF5 file')
    parser.add_argument('--n_pools', type=int, default=1000, help='Maximum number of pools to use')
    
    # Model arguments
    parser.add_argument('--n_lags', type=int, default=5, help='Number of lag steps')
    parser.add_argument('--d_model', type=int, default=32, help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--dense_units', type=int, default=16, help='Dense layer units')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Data split arguments
    parser.add_argument('--train_start', type=str, required=True, help='Training start date')
    parser.add_argument('--train_end', type=str, required=True, help='Training end date')
    parser.add_argument('--val_start', type=str, help='Validation start date')
    parser.add_argument('--val_end', type=str, help='Validation end date')
    parser.add_argument('--test_start', type=str, help='Test start date')
    parser.add_argument('--test_end', type=str, help='Test end date')
    parser.add_argument('--use_validation', action='store_true', help='Use validation set')
    
    # Model saving
    parser.add_argument('--model_name', type=str, default='hybrid_transformer_pretrained', help='Model name')
    parser.add_argument('--features', type=str, required=True, help='Comma-separated feature names')
    parser.add_argument('--targets', type=str, required=True, help='Comma-separated target names')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Initialize distributed training
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("[run_ddp_hybrid_pretraining.py] Configuration:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        sys.stdout.flush()

    # Parse features and targets
    features = [f.strip() for f in args.features.split(',')]
    targets = [t.strip() for t in args.targets.split(',')]
    
    if len(targets) != 2:
        raise ValueError("Hybrid model requires exactly 2 targets: [liquidity_return, volume_return]")
    
    # Prepare split dates
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }
    
    if args.use_validation:
        split_dates.update({
            'val_start': args.val_start,
            'val_end': args.val_end
        })

    print("Preparing dataset...")
    
    # Load available pools
    import h5py
    with h5py.File(args.input_hdf5, 'r') as h5f:
        all_pools = [key.replace('pool_', '') for key in h5f.keys() if key.startswith('pool_')]
    
    print(f"Number of pools in HDF5: {len(all_pools)}")
    
    # Limit to n_pools
    if len(all_pools) > args.n_pools:
        selected_pools = all_pools[:args.n_pools]
    else:
        selected_pools = all_pools
    
    # Distribute pools across processes
    pools_per_process = len(selected_pools) // world_size
    start_idx = rank * pools_per_process
    if rank == world_size - 1:
        end_idx = len(selected_pools)
    else:
        end_idx = start_idx + pools_per_process
    
    process_pools = selected_pools[start_idx:end_idx]
    print(f"Process {rank}: Using pools {start_idx}-{end_idx-1} ({len(process_pools)} pools)")
    
    # Fit scalers on rank 0 only
    if rank == 0:
        print("Fitting scalers...")
        feature_scaler, target_reg_scalers = fit_scalers(
            args.input_hdf5,
            pool_addresses=selected_pools[:1000],  # Use subset for fitting
            features=features,
            targets=targets,
            split='train',
            split_dates=split_dates,
            sample_rate=0.1,
            verbose=1
        )
        # Save scalers
        scaler_path = f"python/ml/PLV/models/{args.model_name}_scalers.pkl"
        save_scalers(feature_scaler, target_reg_scalers, scaler_path)
    else:
        feature_scaler = None
        target_reg_scalers = None
    
    # Broadcast scalers to all processes
    if rank == 0:
        scaler_data = (feature_scaler, target_reg_scalers)
    else:
        scaler_data = None
    
    # Simple broadcast using object_list
    scaler_list = [scaler_data]
    dist.broadcast_object_list(scaler_list, src=0)
    feature_scaler, target_reg_scalers = scaler_list[0]
    
    if rank == 0:
        print("Scalers broadcast to all ranks.")

    # Load datasets
    print("Loading dataset...")
    train_dataset = HybridLPsDataset(
        hdf5_path=args.input_hdf5,
        pool_addresses=process_pools,
        features=features,
        targets=targets,
        n_lags=args.n_lags,
        split='train',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers,
        verbose=1
    )
    
    if args.use_validation:
        val_dataset = HybridLPsDataset(
            hdf5_path=args.input_hdf5,
            pool_addresses=process_pools,
            features=features,
            targets=targets,
            n_lags=args.n_lags,
            split='val',
            split_dates=split_dates,
            feature_scaler=feature_scaler,
            target_reg_scalers=target_reg_scalers,
            verbose=1
        )
    else:
        val_dataset = None

    # Print dataset info
    print(f"[rank {rank}]  Local samples - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Pools: {len(process_pools)}")
    
    # Gather total sample counts
    train_count = torch.tensor(len(train_dataset), device=device)
    val_count = torch.tensor(len(val_dataset) if val_dataset else 0, device=device)
    pool_count = torch.tensor(len(process_pools), device=device)
    
    dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(pool_count, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print(f"Total training samples across all processes: {train_count.item()}")
        print(f"Total validation samples across all processes: {val_count.item()}")
        print(f"Total number of pools across all processes: {pool_count.item()}")
        print(f"Number of features: {len(features)}")
        print(f"Using validation: {args.use_validation}")

    # Create data loaders
    print("Creating DataLoaders for Hybrid Dataset with dataset-level sharding...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    else:
        val_loader = None

    # Create model
    input_size = len(features) + len(targets)  # features + target lags
    model = HybridTransformer(
        input_size=input_size,
        n_lags=args.n_lags,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dense_units=args.dense_units,
        dropout=args.dropout
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[device.index])
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            # Hybrid batch: (X, y_cls_1, y_reg_1, y_reg_2)
            X, y_cls_1, y_reg_1, y_reg_2 = batch
            X = X.to(device)
            y_cls_1 = y_cls_1.to(device).unsqueeze(1)
            y_reg_1 = y_reg_1.to(device).unsqueeze(1)
            y_reg_2 = y_reg_2.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            cls_pred_1, reg_pred_1, reg_pred_2 = model(X)
            loss = HybridTransformer.custom_hybrid_loss(
                cls_pred_1, reg_pred_1, y_cls_1, y_reg_1, reg_pred_2, y_reg_2
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        # Average loss across all processes
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_loss_tensor.item() / world_size
        
        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    X, y_cls_1, y_reg_1, y_reg_2 = batch
                    X = X.to(device)
                    y_cls_1 = y_cls_1.to(device).unsqueeze(1)
                    y_reg_1 = y_reg_1.to(device).unsqueeze(1)
                    y_reg_2 = y_reg_2.to(device).unsqueeze(1)
                    
                    cls_pred_1, reg_pred_1, reg_pred_2 = model(X)
                    loss = HybridTransformer.custom_hybrid_loss(
                        cls_pred_1, reg_pred_1, y_cls_1, y_reg_1, reg_pred_2, y_reg_2
                    )
                    
                    total_val_loss += loss.item()
                    n_val_batches += 1
            
            avg_val_loss = total_val_loss / n_val_batches if n_val_batches > 0 else 0
            val_loss_tensor = torch.tensor(avg_val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model on rank 0
                if rank == 0:
                    model_path = f"python/ml/PLV/models/{args.model_name}.pt"
                    arch_config = {
                        'input_size': input_size,
                        'n_lags': args.n_lags,
                        'd_model': args.d_model,
                        'num_heads': args.num_heads,
                        'num_layers': args.num_layers,
                        'dense_units': args.dense_units,
                        'dropout': args.dropout,
                        'features': features,
                        'targets': targets,
                        'model_type': 'hybrid_transformer'
                    }
                    save_hybrid_model(model, model_path, arch_config)
            else:
                patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        
        if rank == 0:
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {val_loss:.12f}, Time: {epoch_time:.2f}s")
            else:
                if epoch % 5 == 0 or epoch == args.epochs - 1:
                    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.12f}, Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if val_loader is not None and patience_counter >= early_stopping_patience:
            if rank == 0:
                print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.12f}")
            break
    
    if rank == 0:
        print("Training complete.")
        if val_loader is None:
            # Save final model if no validation
            model_path = f"python/ml/PLV/models/{args.model_name}.pt"
            arch_config = {
                'input_size': input_size,
                'n_lags': args.n_lags,
                'd_model': args.d_model,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'dense_units': args.dense_units,
                'dropout': args.dropout,
                'features': features,
                'targets': targets,
                'model_type': 'hybrid_transformer'
            }
            save_hybrid_model(model, model_path, arch_config)

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
