#!/usr/bin/env python
"""
Enhanced finetuning script with frozen encoder approach.
Keeps pretrained encoder weights intact and only trains task heads.
"""

import os
import argparse
import torch
import torch.nn as nn
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from python.ml.PLV.data_io import LPsDataset, fit_scalers
from python.ml.PLV.model import ZeroInflatedTransformer
from python.utils.distributed_utils import load_model_arch0, save_model_arch0, load_scalers, save_scalers0

def enhanced_finetune(model, train_loader, val_loader, epochs=50, base_lr=0.0001, 
                     warmup_epochs=5, verbose=True):
    """
    Enhanced finetuning with frozen encoder approach:
    - Freeze encoder, train only task heads  
    - Use warmup for stable training
    - Keep pretrained encoder weights intact
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    best_state = None
    
    print(f"=== Enhanced Finetuning Strategy ===")
    print(f"Freeze encoder, train heads only ({epochs} epochs)")
    print(f"Head LR: {base_lr * 5}, Warmup epochs: {warmup_epochs}")
    
    # Freeze encoder, train only heads
    model.freeze_encoder()
    print("Froze encoder layers for finetuning")
    
    # Only optimize unfrozen parameters (heads)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=base_lr * 5)  # Higher LR for heads only
    
    for epoch in range(epochs):
        # Warmup learning rate in first few epochs
        if epoch < warmup_epochs:
            warmup_lr = base_lr * 5 * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = batch
            X_tensor = X.to(device)
            y_cls_1 = y_cls_1.to(device).unsqueeze(1)
            y_reg_1 = y_reg_1.to(device).unsqueeze(1)
            y_cls_2 = y_cls_2.to(device).unsqueeze(1)
            y_reg_2 = y_reg_2.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = model(X_tensor)
            
            # Multi-task loss
            cls_loss_1 = nn.BCELoss()(cls_pred_1, y_cls_1)
            reg_loss_1 = nn.MSELoss()(reg_pred_1, y_reg_1)
            cls_loss_2 = nn.BCELoss()(cls_pred_2, y_cls_2)
            reg_loss_2 = nn.MSELoss()(reg_pred_2, y_reg_2)
            
            loss = cls_loss_1 + reg_loss_1 + cls_loss_2 + reg_loss_2
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Validation
        if val_loader:
            val_loss = model.evaluate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
            
            if verbose:
                avg_train_loss = total_loss / n_batches if n_batches > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    return model

def main(args):
    # Configuration
    hdf5_path = args.hdf5_path
    finetune_pool_address = args.finetune_pool_address
    pretrained_model_name = args.pretrained_model_name
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    train_start = args.train_start
    train_end = args.train_end
    val_start = args.val_start
    val_end = args.val_end
    
    # Paths
    models_dir = Path("python/ml/PLV/models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{pretrained_model_name}.pt"
    
    # Print configuration
    print(f"=== Enhanced Finetuning Job Starting ===")
    print(f"Pool: {finetune_pool_address}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Training Period: {train_start} to {train_end}")
    print(f"Validation Period: {val_start} to {val_end}")
    
    # Load saved architecture to determine features
    arch = load_model_arch0(model_path)
    features = arch['features']
    targets = arch['targets']
    
    print(f"Fitting scalers on target pool data...")
    # Fit scalers on finetuning data
    feature_scaler, target_reg_scalers = fit_scalers(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        targets=targets,
        fraction=0.1  # Use 10% for scaler fitting
    )
    
    # Create datasets
    train_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        targets=targets,
        start_date=train_start,
        end_date=train_end,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers
    )
    
    val_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        targets=targets,
        start_date=val_start,
        end_date=val_end,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load pretrained model
    print(f"Loading pretrained model: {pretrained_model_name}")
    model_kwargs = {k: v for k, v in arch.items() if k not in ['features', 'targets']}
    model_kwargs['input_size'] = len(features) + len(targets)
    
    model = ZeroInflatedTransformer(**model_kwargs)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle both distributed and regular checkpoint formats
    if 'model' in checkpoint:
        # Distributed training format: {'model': state_dict, 'optimizer': optim_state}
        model.load_state_dict(checkpoint['model'])
    else:
        # Regular format: state_dict directly
        model.load_state_dict(checkpoint)
    
    print("Starting enhanced finetuning...")
    model = enhanced_finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        base_lr=lr,
        verbose=True
    )
    
    # Save finetuned model
    finetuned_name = f"{pretrained_model_name}_finetuned_{finetune_pool_address[:10]}"
    finetuned_path = models_dir / f"{finetuned_name}.pt"
    finetuned_scalers_path = models_dir / f"{finetuned_name}_scalers.pkl"
    
    torch.save(model.state_dict(), finetuned_path)
    save_model_arch0(finetuned_path, **arch)
    save_scalers0(feature_scaler, target_reg_scalers, finetuned_scalers_path)
    
    print(f"=== Enhanced Finetuning Complete ===")
    print(f"Saved finetuned model: {finetuned_path}")
    print(f"Saved scalers: {finetuned_scalers_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced finetuning for Zero-Inflated Transformer")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--finetune_pool_address", type=str, required=True, help="Pool address for finetuning")
    parser.add_argument("--pretrained_model_name", type=str, default="transformer_pretrained", help="Name of pretrained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_start", type=str, default="2023-01-01", help="Training start date")
    parser.add_argument("--train_end", type=str, default="2025-05-01", help="Training end date")
    parser.add_argument("--val_start", type=str, default="2025-05-02", help="Validation start date") 
    parser.add_argument("--val_end", type=str, default="2025-06-01", help="Validation end date")
    
    args = parser.parse_args()
    main(args)
