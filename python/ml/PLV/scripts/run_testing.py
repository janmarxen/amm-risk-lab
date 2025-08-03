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
    target = arch['target']
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
    feature_scaler, target_reg_scaler = load_scalers(scaler_path)
    print("Preparing test dataset...")
    test_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[pool_address],
        features=features,
        target=target,
        n_lags=arch['n_lags'],
        split='test',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scaler=target_reg_scaler,
        verbose=1
    )
    if len(test_dataset) == 0:
        print("No test data available.")
        return
        # Compare scaler stats with test data
    print("Feature scaler mean:", feature_scaler.mean_)
    print("Feature scaler std:", feature_scaler.scale_)
    print("Test feature mean:", np.mean(test_dataset.X.numpy(), axis=(0, 1)))
    print("Test feature std:", np.std(test_dataset.X.numpy(), axis=(0, 1)))
    print("Target scaler mean:", target_reg_scaler.mean_)
    print("Target scaler std:", target_reg_scaler.scale_)
    print("Test target mean:", np.mean(test_dataset.y_reg.numpy()))
    print("Test target std:", np.std(test_dataset.y_reg.numpy()))
    # --- Load model ---
    print("Loading model...")
    input_size = len(features) + 1 # +1 for the target variable
    model = ZeroInflatedTransformer(
        input_size=input_size,
        n_lags=arch['n_lags'],
        d_model=arch['d_model'],
        num_heads=arch['num_heads'],
        num_layers=arch['num_layers'],
        dense_units=arch['dense_units'],
        dropout=arch['dropout']
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # --- Model predictions ---
    X = test_dataset.X
    y_cls = test_dataset.y_cls
    y_reg = test_dataset.y_reg
    print("Unique y_cls in test set:", np.unique(y_cls.numpy(), return_counts=True))
    y_reg_pred, y_cls_pred = model.predict(X)
    # Print number of zero class predictions
    n_zero_pred = np.sum(y_cls_pred == 1)
    n_total = len(y_cls_pred)
    print(f"Zero class predictions: {n_zero_pred} out of {n_total} ({n_zero_pred/n_total:.2%})")

    # --- Custom loss on test set ---
    print("Calculating custom loss on test set...")
    with torch.no_grad():
        y_cls_tensor = y_cls.unsqueeze(1)
        y_reg_tensor = y_reg.unsqueeze(1)
        cls_pred, reg_pred = model(X)
        test_loss = model.__class__.custom_zi_loss(cls_pred, reg_pred, y_cls_tensor, y_reg_tensor).item()
    print(f"Model's test custom loss: {test_loss:.8f}")
    
    # --- Naive baseline ---
    print("Calculating naive baseline...")
    y_reg_np = y_reg.numpy()
    naive_pred = naive_predict(np.array(y_reg_np))
    # For custom loss, need to align shapes and mask
    mask = ~np.isnan(naive_pred)
    y_reg_tensor_naive = torch.tensor(naive_pred[mask], dtype=torch.float32)
    y_cls_tensor_naive = y_cls[mask].unsqueeze(1)
    y_reg_tensor_true = y_reg[mask]
    # Naive loss: use true y_cls, naive y_reg
    naive_loss = model.__class__.custom_zi_loss(y_cls_tensor_naive, y_reg_tensor_naive.unsqueeze(1), y_cls_tensor_naive, y_reg_tensor_true.unsqueeze(1)).item()
    print(f"Naive baseline custom loss: {naive_loss:.8f}")
    # --- Plot ---
    print("Saving figures...")
    y_reg_pred[y_cls_pred==1] = 0  # Set predicted liquidity return to 0 where cls_pred is 1
    save_actual_vs_predicted(y_reg_np, y_reg_pred, title=f"Actual vs Predicted {model_name} (Test Set)", filename=f"actual_vs_predicted_{model_name}.png")
    save_actual_vs_predicted(y_reg_np[mask], naive_pred[mask], title=f"Naive: Actual vs Predicted {model_name} (Test Set)", filename=f"naive_actual_vs_predicted_{model_name}.png")


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

