"""
run_testing.py

Testing script for evaluating finetuned Uniswap V3 ML models on test data.

High-level steps:
1. Parse command-line arguments for model configuration and test pool.
2. Load model architecture and configuration from saved JSON file.
3. Load the fitted scalers that were saved during finetuning to ensure
   consistent scaling between training and testing phases.
4. Construct test dataset for the specified pool using the loaded scalers
   to maintain scaling consistency and prevent data leakage.
5. Load the finetuned model from checkpoint and set to evaluation mode.
6. Generate predictions on the test set using the finetuned model.
7. Calculate model performance metrics including custom zero-inflated loss.
8. Compare against naive persistence baseline for performance evaluation.
9. Generate and save visualization plots of actual vs predicted values.

Note: This script loads scalers fitted during finetuning to ensure consistent
scaling across train/val/test splits and prevent any data leakage during evaluation.
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
    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
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
    # scaler_path = os.path.splitext(model_path)[0] + '_scalers.pkl'
    scaler_path = 'python/ml/PLV/models/transformer_multi_task_finetuned_1_0xcbcdf9626bc03e24f779434178a73a0b4bad62ed_scalers.pkl'
    feature_scaler, target_reg_scalers = load_scalers(scaler_path)
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
    # --- Load model ---
    print("Loading model...")
    input_size = len(features) * len(targets) + len(targets)  # Multi-task input size
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
    y_cls_2 = test_dataset.y_cls_2
    y_reg_2 = test_dataset.y_reg_2
    print("Unique y_cls_1 in test set:", np.unique(y_cls_1.numpy(), return_counts=True))
    print("Unique y_cls_2 in test set:", np.unique(y_cls_2.numpy(), return_counts=True))
    y_reg_pred_1, y_cls_pred_1, y_reg_pred_2, y_cls_pred_2 = model.predict(X)
    # Print number of zero class predictions for both tasks
    n_zero_pred_1 = np.sum(y_cls_pred_1 == 1)
    n_zero_pred_2 = np.sum(y_cls_pred_2 == 1)
    n_total = len(y_cls_pred_1)
    print(f"Task 1 zero class predictions: {n_zero_pred_1} out of {n_total} ({n_zero_pred_1/n_total:.2%})")
    print(f"Task 2 zero class predictions: {n_zero_pred_2} out of {n_total} ({n_zero_pred_2/n_total:.2%})")

    # --- Custom loss on test set ---
    print("Calculating custom loss on test set...")
    with torch.no_grad():
        y_cls_tensor_1 = y_cls_1.unsqueeze(1)
        y_reg_tensor_1 = y_reg_1.unsqueeze(1)
        y_cls_tensor_2 = y_cls_2.unsqueeze(1)
        y_reg_tensor_2 = y_reg_2.unsqueeze(1)
        cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = model(X)
        test_loss = model.__class__.custom_zi_loss(cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2, y_cls_tensor_1, y_reg_tensor_1, y_cls_tensor_2, y_reg_tensor_2).item()
    print(f"Model's test custom loss: {test_loss:.8f}")
    
    # --- Naive baseline ---
    print("Calculating naive baseline...")
    y_reg_np_1 = y_reg_1.numpy()
    y_reg_np_2 = y_reg_2.numpy()
    naive_pred_1 = naive_predict(np.array(y_reg_np_1))
    naive_pred_2 = naive_predict(np.array(y_reg_np_2))
    # For custom loss, need to align shapes and mask
    mask = ~np.isnan(naive_pred_1) & ~np.isnan(naive_pred_2)
    y_reg_tensor_naive_1 = torch.tensor(naive_pred_1[mask], dtype=torch.float32)
    y_reg_tensor_naive_2 = torch.tensor(naive_pred_2[mask], dtype=torch.float32)
    y_cls_tensor_naive_1 = y_cls_1[mask].unsqueeze(1)
    y_cls_tensor_naive_2 = y_cls_2[mask].unsqueeze(1)
    y_reg_tensor_true_1 = y_reg_1[mask]
    y_reg_tensor_true_2 = y_reg_2[mask]
    # Naive loss: use true y_cls, naive y_reg
    naive_loss = model.__class__.custom_zi_loss(y_cls_tensor_naive_1, y_reg_tensor_naive_1.unsqueeze(1), y_cls_tensor_naive_2, y_reg_tensor_naive_2.unsqueeze(1), y_cls_tensor_naive_1, y_reg_tensor_true_1.unsqueeze(1), y_cls_tensor_naive_2, y_reg_tensor_true_2.unsqueeze(1)).item()
    print(f"Naive baseline custom loss: {naive_loss:.8f}")
    # --- Plot ---
    print("Saving figures...")
    # Task 1 plots
    y_reg_pred_1[y_cls_pred_1==1] = 0  # Set predicted values to 0 where cls_pred is 1
    save_actual_vs_predicted(y_reg_np_1, y_reg_pred_1, title=f"Actual vs Predicted {targets[0]} {model_name} (Test Set)", filename=f"actual_vs_predicted_{model_name}_{targets[0]}.png")
    save_actual_vs_predicted(y_reg_np_1[mask], naive_pred_1[mask], title=f"Naive: Actual vs Predicted {targets[0]} {model_name} (Test Set)", filename=f"naive_actual_vs_predicted_{model_name}_{targets[0]}.png")
    
    # Task 2 plots
    y_reg_pred_2[y_cls_pred_2==1] = 0  # Set predicted values to 0 where cls_pred is 1
    save_actual_vs_predicted(y_reg_np_2, y_reg_pred_2, title=f"Actual vs Predicted {targets[1]} {model_name} (Test Set)", filename=f"actual_vs_predicted_{model_name}_{targets[1]}.png")
    save_actual_vs_predicted(y_reg_np_2[mask], naive_pred_2[mask], title=f"Naive: Actual vs Predicted {targets[1]} {model_name} (Test Set)", filename=f"naive_actual_vs_predicted_{model_name}_{targets[1]}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--pool_address', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

