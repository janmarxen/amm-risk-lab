#!/usr/bin/env python3
"""
Distributed finetuning script for Hybrid Multi-task Learning:
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
from torch.utils.data import DataLoader

sys.path.append('/p/project1/training2529/marxen1/amm-risk-lab')
from python.ml.PLV.data_io import load_scalers, HybridLPsDataset
from python.ml.PLV.model import HybridTransformer

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    return rank, world_size, device

def load_hybrid_model(model_path, device):
    """Load hybrid model from checkpoint."""
    # Load architecture
    arch_path = os.path.splitext(model_path)[0] + '_arch.json'
    with open(arch_path, 'r') as f:
        arch = json.load(f)
    
    # Create model
    model = HybridTransformer(
        input_size=arch['input_size'],
        n_lags=arch['n_lags'],
        d_model=arch['d_model'],
        num_heads=arch['num_heads'],
        num_layers=arch['num_layers'],
        dense_units=arch['dense_units'],
        dropout=arch['dropout']
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    return model, arch

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
    parser = argparse.ArgumentParser(description='Distributed Hybrid Finetuning')
    
    # Data arguments
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--finetune_pool_address', type=str, required=True, help='Pool address for finetuning')
    
    # Model arguments
    parser.add_argument('--pretrained_model_name', type=str, required=True, help='Pretrained model name')
    parser.add_argument('--finetuned_model_name', type=str, required=True, help='Finetuned model name')
    
    # Training arguments
    parser.add_argument('--finetune_epochs', type=int, default=10, help='Number of finetuning epochs')
    parser.add_argument('--finetune_batch_size', type=int, default=64, help='Finetuning batch size')
    parser.add_argument('--finetune_lr', type=float, default=0.0005, help='Finetuning learning rate')
    
    # Data split arguments
    parser.add_argument('--train_start', type=str, required=True, help='Training start date')
    parser.add_argument('--train_end', type=str, required=True, help='Training end date')
    parser.add_argument('--val_start', type=str, help='Validation start date')
    parser.add_argument('--val_end', type=str, help='Validation end date')
    parser.add_argument('--test_start', type=str, help='Test start date')
    parser.add_argument('--test_end', type=str, help='Test end date')
    parser.add_argument('--use_validation', action='store_true', help='Use validation set')
    
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
        print("[run_ddp_hybrid_finetuning.py] Configuration:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        sys.stdout.flush()

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

    print(f"Using validation: {args.use_validation}")
    
    # Load pretrained model
    print("Loading model...")
    pretrained_path = f"python/ml/PLV/models/{args.pretrained_model_name}.pt"
    model, arch = load_hybrid_model(pretrained_path, device)
    
    # Load scalers on rank 0 and broadcast
    if rank == 0:
        print("Fitting scalers on rank 0...")
        scaler_path = f"python/ml/PLV/models/{args.pretrained_model_name}_scalers.pkl"
        feature_scaler, target_reg_scalers = load_scalers(scaler_path)
        scaler_data = (feature_scaler, target_reg_scalers)
    else:
        scaler_data = None
    
    # Broadcast scalers
    scaler_list = [scaler_data]
    dist.broadcast_object_list(scaler_list, src=0)
    feature_scaler, target_reg_scalers = scaler_list[0]
    
    if rank == 0:
        print("Scalers loaded and broadcasted to all ranks.")
    
    # Create finetuning dataset
    print("Finetuning model on test pool on training+validation dates...")
    train_dataset = HybridLPsDataset(
        hdf5_path=args.hdf5_path,
        pool_addresses=[args.finetune_pool_address],
        features=arch['features'],
        targets=arch['targets'],
        n_lags=arch['n_lags'],
        split='train',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers,
        verbose=1
    )
    
    if args.use_validation and 'val_start' in split_dates and 'val_end' in split_dates:
        val_dataset = HybridLPsDataset(
            hdf5_path=args.hdf5_path,
            pool_addresses=[args.finetune_pool_address],
            features=arch['features'],
            targets=arch['targets'],
            n_lags=arch['n_lags'],
            split='val',
            split_dates=split_dates,
            feature_scaler=feature_scaler,
            target_reg_scalers=target_reg_scalers,
            verbose=1
        )
    else:
        val_dataset = None
    
    print(f"Starting finetuning...")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.finetune_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    else:
        val_loader = None

    # Wrap with DDP
    model = DDP(model, device_ids=[device.index])
    
    # Finetuning loop
    print("Finetuning model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    
    for epoch in range(args.finetune_epochs):
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
        
        epoch_time = time.time() - epoch_start_time
        
        if rank == 0:
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{args.finetune_epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {val_loss:.12f}, Time: {epoch_time:.2f}s")
            else:
                if epoch % 5 == 0 or epoch == args.finetune_epochs - 1:
                    print(f"Epoch {epoch+1}/{args.finetune_epochs}, Train Loss: {avg_train_loss:.12f}, Time: {epoch_time:.2f}s")
    
    # Save finetuned model on rank 0
    if rank == 0:
        print("Finetuning complete.")
        finetuned_path = f"python/ml/PLV/models/{args.finetuned_model_name}.pt"
        arch_config = arch.copy()
        arch_config['finetune_pool'] = args.finetune_pool_address
        save_hybrid_model(model, finetuned_path, arch_config)
        
        # Copy scalers
        import shutil
        src_scaler_path = f"python/ml/PLV/models/{args.pretrained_model_name}_scalers.pkl"
        dst_scaler_path = f"python/ml/PLV/models/{args.finetuned_model_name}_scalers.pkl"
        shutil.copy2(src_scaler_path, dst_scaler_path)
        print(f"Scalers copied to {dst_scaler_path}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
