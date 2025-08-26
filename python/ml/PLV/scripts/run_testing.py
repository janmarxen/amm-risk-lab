"""
run_testing.py

Testing script for evaluating hybrid Uniswap V3 ML models on test data.
- Task 1 (liquidity_return): Zero-inflated modeling (classification + regression)
- Task 2 (volume_return): Standard regression only

High-level steps:
1. Parse command-line arguments for model configuration and test pool.
2. Load model architecture and configuration from saved JSON file.
3. Load the fitted scalers that were saved during finetuning to ensure
   consistent scaling between training and testing phases.
4. Construct test dataset for the specified pool using the loaded scalers
   to maintain scaling consistency and prevent data leakage.
5. Load the hybrid model from checkpoint and set to evaluation mode.
6. Generate predictions on the test set using the hybrid model.
7. Calculate model performance metrics including hybrid loss function.
8. Compare against naive persistence baseline for performance evaluation.
9. Generate and save visualization plots of actual vs predicted values.

Note: This script loads scalers fitted during finetuning to ensure consistent
scaling across train/val/test splits and prevent any data leakage during evaluation.
The hybrid model uses zero-inflated approach for liquidity returns and standard
regression for volume returns, reflecting the different data characteristics.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import json
from python.ml.PLV.data_io import LPsDataset
from python.ml.PLV.model import ZeroInflatedTransformer
from python.utils.distributed_utils import load_scalers


# Baseline naive prediction
def naive_predict(series):
    """Naive model: predicts next value as the current value (persistence)."""
    if hasattr(series, 'shift'):
        return series.shift(1)
    else:
        arr = np.asarray(series)
        result = np.empty_like(arr)
        result[0] = np.nan
        result[1:] = arr[:-1]
        return result

def save_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Liquidity Return", filename="actual_vs_predicted.png"):
    plt.figure(figsize=(14, 5))
    if hasattr(y_true, 'index'):
        x = y_true.index
    else:
        x = np.arange(len(y_true))
    plt.plot(x, y_true, label="Actual", color="tab:blue")
    plt.plot(x, y_pred, label="Predicted", color="tab:orange")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main(args):
    
    print("[run_testing.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    pool_address = args.pool_address
    model_name = args.model_name
    hdf5_path = args.hdf5_path
    model_path = os.path.join("python/ml/PLV/models", f"{model_name}.pt")
    arch_path = os.path.splitext(model_path)[0] + '_arch.json'
    with open(arch_path, 'r') as f:
        arch = json.load(f)
    features = arch['features']
    targets = arch['targets']  # Multi-task targets
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }
    # --- Prepare test dataset ---
    # Load scalers
    scaler_path = os.path.splitext(model_path)[0] + '_scalers.pkl'
    feature_scaler, target_reg_scalers = load_scalers(scaler_path)
    
    # Debug: Print loaded scaler statistics
    print("=== LOADED SCALER DEBUG INFO ===")
    print(f"Feature scaler: {type(feature_scaler).__name__}")
    if hasattr(feature_scaler, 'mean_') and feature_scaler.mean_ is not None:
        print(f"  Features mean: {feature_scaler.mean_[:5]}... (showing first 5)")
        print(f"  Features std: {feature_scaler.scale_[:5]}... (showing first 5)")
    
    print(f"Target 1 ({targets[0]}) scaler: {type(target_reg_scalers[0]).__name__}")
    if hasattr(target_reg_scalers[0], 'nonzero_mean_'):
        print(f"  {targets[0]} nonzero_mean: {target_reg_scalers[0].nonzero_mean_}")
        print(f"  {targets[0]} nonzero_std: {target_reg_scalers[0].nonzero_std_}")
    elif hasattr(target_reg_scalers[0], 'mean_'):
        print(f"  {targets[0]} mean: {target_reg_scalers[0].mean_}")
        print(f"  {targets[0]} std: {target_reg_scalers[0].scale_}")
    
    print(f"Target 2 ({targets[1]}) scaler: {type(target_reg_scalers[1]).__name__}")
    if hasattr(target_reg_scalers[1], 'nonzero_mean_'):
        print(f"  {targets[1]} nonzero_mean: {target_reg_scalers[1].nonzero_mean_}")
        print(f"  {targets[1]} nonzero_std: {target_reg_scalers[1].nonzero_std_}")
    elif hasattr(target_reg_scalers[1], 'mean_'):
        print(f"  {targets[1]} mean: {target_reg_scalers[1].mean_}")
        print(f"  {targets[1]} std: {target_reg_scalers[1].scale_}")
    elif hasattr(target_reg_scalers[1], 'median_'):
        print(f"  {targets[1]} median: {target_reg_scalers[1].median_}")
        print(f"  {targets[1]} scale: {target_reg_scalers[1].scale_}")
    print("===============================")
    
    print("Preparing test dataset...")
    test_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[pool_address],
        features=features,
        targets=targets,
        n_lags=arch['n_lags'],
        split='test',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers,
        verbose=1
    )
    if len(test_dataset) == 0:
        print("No test data available.")
        return
    
    # Debug: Print raw data statistics
    print("=== RAW TEST DATA DEBUG INFO ===")
    y_reg_1_np = test_dataset.y_reg_1.numpy()
    y_reg_2_np = test_dataset.y_reg_2.numpy()
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"{targets[0]} (scaled) stats:")
    print(f"  Mean: {np.mean(y_reg_1_np):.6f}")
    print(f"  Std: {np.std(y_reg_1_np):.6f}")
    print(f"  Min: {np.min(y_reg_1_np):.6f}")
    print(f"  Max: {np.max(y_reg_1_np):.6f}")
    print(f"  Zeros: {np.sum(y_reg_1_np == 0)} / {len(y_reg_1_np)} ({np.sum(y_reg_1_np == 0)/len(y_reg_1_np):.2%})")
    
    print(f"{targets[1]} (scaled) stats:")
    print(f"  Mean: {np.mean(y_reg_2_np):.6f}")
    print(f"  Std: {np.std(y_reg_2_np):.6f}")
    print(f"  Min: {np.min(y_reg_2_np):.6f}")
    print(f"  Max: {np.max(y_reg_2_np):.6f}")
    print(f"  Zeros: {np.sum(y_reg_2_np == 0)} / {len(y_reg_2_np)} ({np.sum(y_reg_2_np == 0)/len(y_reg_2_np):.2%})")
    
    # **CRITICAL DEBUG: Check if volume_return has true zeros or just small values**
    print(f"  Exact zeros: {np.sum(y_reg_2_np == 0.0)}")
    print(f"  Very small values (< 1e-10): {np.sum(np.abs(y_reg_2_np) < 1e-10)}")
    print(f"  Near-zero values (< 0.001): {np.sum(np.abs(y_reg_2_np) < 0.001)}")
    
    print("===============================")
    
    print("Loading model...")
    input_size = len(features) + len(targets)  # Correct input size: features + target lags
    model = ZeroInflatedTransformer(
        input_size=input_size,
        n_lags=arch['n_lags'],
        d_model=arch['d_model'],
        num_heads=arch['num_heads'],
        num_layers=arch['num_layers'],
        dense_units=arch['dense_units'],
        dropout=arch['dropout']
    )
    
    # Handle both finetuned and pretrained model formats
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Pretrained model format (from save_full_model)
        model.load_state_dict(checkpoint['model'])
    else:
        # Finetuned model format (from save0) - direct state_dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    # --- Model predictions ---
    X = test_dataset.X
    y_cls_1 = test_dataset.y_cls_1
    y_reg_1 = test_dataset.y_reg_1
    y_reg_2 = test_dataset.y_reg_2
    print("Unique y_cls_1 in test set:", np.unique(y_cls_1.numpy(), return_counts=True))
    y_reg_pred_1, y_cls_pred_1, y_reg_pred_2 = model.predict(X)
    
    # **FIX: Inverse transform predictions back to original scale**
    print("Inverse transforming predictions back to original scale...")
    
    # Task 1 (liquidity_return): Apply zero-preserving inverse transform
    y_reg_pred_1_unscaled = target_reg_scalers[0].inverse_transform(y_reg_pred_1.reshape(-1, 1)).flatten()
    
    # Task 2 (volume_return): Apply standard inverse transform  
    y_reg_pred_2_unscaled = target_reg_scalers[1].inverse_transform(y_reg_pred_2.reshape(-1, 1)).flatten()
    
    # Also inverse transform the actual values for comparison
    y_reg_1_unscaled = target_reg_scalers[0].inverse_transform(y_reg_1.numpy().reshape(-1, 1)).flatten()
    y_reg_2_unscaled = target_reg_scalers[1].inverse_transform(y_reg_2.numpy().reshape(-1, 1)).flatten()
    
    print(f"Predictions inverse transformed:")
    print(f"  Task 1 (liquidity) pred range: [{np.min(y_reg_pred_1_unscaled):.3f}, {np.max(y_reg_pred_1_unscaled):.3f}]")
    print(f"  Task 2 (volume) pred range: [{np.min(y_reg_pred_2_unscaled):.3f}, {np.max(y_reg_pred_2_unscaled):.3f}]")
    print(f"  Task 1 (liquidity) actual range: [{np.min(y_reg_1_unscaled):.3f}, {np.max(y_reg_1_unscaled):.3f}]")
    print(f"  Task 2 (volume) actual range: [{np.min(y_reg_2_unscaled):.3f}, {np.max(y_reg_2_unscaled):.3f}]")
    
    # Print number of zero class predictions for Task 1 only (Task 2 is standard regression)
    n_zero_pred_1 = np.sum(y_cls_pred_1 == 1)
    n_total = len(y_cls_pred_1)
    print(f"Task 1 (liquidity) zero class predictions: {n_zero_pred_1} out of {n_total} ({n_zero_pred_1/n_total:.2%})")
    print(f"Task 2 (volume) uses standard regression (no zero-class prediction)")

    # --- Custom loss on test set ---
    print("Calculating custom loss on test set...")
    with torch.no_grad():
        y_cls_tensor_1 = y_cls_1.unsqueeze(1)
        y_reg_tensor_1 = y_reg_1.unsqueeze(1)
        y_reg_tensor_2 = y_reg_2.unsqueeze(1)
        cls_pred_1, reg_pred_1, reg_pred_2 = model(X)
        test_loss = model.__class__.custom_zi_loss(
            cls_pred_1, reg_pred_1, y_cls_tensor_1, y_reg_tensor_1,
            reg_pred_2, y_reg_tensor_2
        ).item()
    print(f"Model's test custom loss: {test_loss:.8f}")
    
    # --- Naive baseline ---
    print("Calculating naive baseline...")
    # Use unscaled values for naive baseline
    naive_pred_1 = naive_predict(np.array(y_reg_1_unscaled))
    naive_pred_2 = naive_predict(np.array(y_reg_2_unscaled))
    
    # For custom loss, need to use scaled values (since model expects scaled targets)
    mask = ~np.isnan(naive_pred_1) & ~np.isnan(naive_pred_2)
    
    # Scale the naive predictions back for loss calculation
    naive_pred_1_scaled = target_reg_scalers[0].transform(naive_pred_1[mask].reshape(-1, 1)).flatten()
    naive_pred_2_scaled = target_reg_scalers[1].transform(naive_pred_2[mask].reshape(-1, 1)).flatten()
    
    # Create naive classification predictions
    # Task 1: Assume all non-zero for naive baseline  
    # Task 2: Standard regression only (no classification component)
    naive_cls_pred_1 = torch.zeros(len(naive_pred_1_scaled), 1)  # Assume all non-zero class
    # Create naive regression tensors
    y_reg_tensor_naive_1 = torch.tensor(naive_pred_1_scaled, dtype=torch.float32).unsqueeze(1)
    y_reg_tensor_naive_2 = torch.tensor(naive_pred_2_scaled, dtype=torch.float32).unsqueeze(1)
    y_cls_tensor_naive_1 = y_cls_1[mask].unsqueeze(1)
    y_reg_tensor_true_1 = y_reg_1[mask].unsqueeze(1)
    y_reg_tensor_true_2 = y_reg_2[mask].unsqueeze(1)
    
    # Naive loss: use naive predictions vs true targets
    naive_loss = model.__class__.custom_zi_loss(
        naive_cls_pred_1, y_reg_tensor_naive_1, y_cls_tensor_naive_1, y_reg_tensor_true_1,
        y_reg_tensor_naive_2, y_reg_tensor_true_2
    ).item()
    print(f"Naive baseline custom loss: {naive_loss:.8f}")
    # --- Plot ---
    print("Saving figures...")
    
    # Task 1: Zero-inflated predictions (liquidity_return)
    final_pred_1 = np.where(y_cls_pred_1 == 1, 0.0, y_reg_pred_1_unscaled)
    save_actual_vs_predicted(y_reg_1_unscaled, final_pred_1, 
                           title=f"Actual vs Predicted {targets[0]} {model_name} (Test Set)", 
                           filename=f"actual_vs_predicted_{model_name}_{targets[0]}.png")
    save_actual_vs_predicted(y_reg_1_unscaled[mask], naive_pred_1[mask], 
                           title=f"Naive: Actual vs Predicted {targets[0]} {model_name} (Test Set)", 
                           filename=f"naive_actual_vs_predicted_{model_name}_{targets[0]}.png")
    
    # Task 2: Standard regression predictions (volume_return)
    # No zero-inflated logic needed for Task 2
    save_actual_vs_predicted(y_reg_2_unscaled, y_reg_pred_2_unscaled, 
                           title=f"Actual vs Predicted {targets[1]} {model_name} (Test Set)", 
                           filename=f"actual_vs_predicted_{model_name}_{targets[1]}.png")
    save_actual_vs_predicted(y_reg_2_unscaled[mask], naive_pred_2[mask], 
                           title=f"Naive: Actual vs Predicted {targets[1]} {model_name} (Test Set)", 
                           filename=f"naive_actual_vs_predicted_{model_name}_{targets[1]}.png")
    
    # Additional analysis: print classification accuracy for Task 1 only
    print("\nHybrid Model Analysis:")
    y_cls_true_1 = y_cls_1.numpy()
    
    cls_acc_1 = np.mean(y_cls_pred_1 == y_cls_true_1)
    print(f"Task 1 (liquidity) classification accuracy: {cls_acc_1:.4f}")
    print(f"Task 2 (volume) uses standard regression only")
    
    # Regression accuracy using unscaled values
    # Task 1: Only on non-zero cases
    non_zero_mask_1 = y_cls_true_1 == 0  # Non-zero class
    if np.any(non_zero_mask_1):
        reg_mse_1 = np.mean((y_reg_pred_1_unscaled[non_zero_mask_1] - y_reg_1_unscaled[non_zero_mask_1])**2)
        print(f"Task 1 (liquidity) regression MSE (non-zero cases): {reg_mse_1:.6f}")
    
    # Task 2: All cases (standard regression)
    reg_mse_2 = np.mean((y_reg_pred_2_unscaled - y_reg_2_unscaled)**2)
    print(f"Task 2 (volume) regression MSE (all cases): {reg_mse_2:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--hdf5_path', type=str, required=True)
    parser.add_argument('--pool_address', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

