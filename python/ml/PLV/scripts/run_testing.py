import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from python.ml.PLV.data_io import LPsDataset, fetch_and_save_pools
from python.ml.PLV.model import ZeroInflatedLSTM, ZeroInflatedTransformer
import argparse
import sys

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
    plt.ylabel("Liquidity Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lags', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lstm_units', type=int, required=False, help='Number of LSTM units (required for LSTM)')
    parser.add_argument('--dense_units', type=int, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--subgraph_id', type=str, required=True)
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--main_pool_address', type=str, required=True)
    parser.add_argument('--finetune_epochs', type=int, required=False, default=15)
    parser.add_argument('--finetune_lr', type=float, required=False, default=0.001)
    parser.add_argument('--finetune_batch_size', type=int, required=False, default=32)
    parser.add_argument('--features', type=str, required=False, default=None, help='Comma-separated list of features')
    parser.add_argument('--target', type=str, required=False, default=None, help='Target column name')
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'], help='Model type to use')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    # Check required arguments for LSTM
    if args.model_type == "lstm" and args.lstm_units is None:
        parser.error("--lstm_units is required when --model_type is 'lstm'")

    print("[run_testing.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    main_pool_address = args.main_pool_address
    model_name = args.model_name
    # --- Config ---
    # hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    model_path = os.path.join("python/ml/PLV/models", f"{model_name}.pt")
    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print("[run_testing.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.target is not None:
        target = args.target
    else:
        print("[run_testing.py] ERROR: --target argument must be specified.")
        sys.exit(1)
    api_key = args.api_key
    subgraph_id = args.subgraph_id
    start_date = args.train_start  # Use train_start for data download
    end_date = args.test_end       # Use test_end for data download
    min_rows = 50
    hdf5_mode = 'x'  # 'x' = fail if exists
    # print("Downloading test pool data...")
    # fetch_and_save_pools(
    #     api_key=api_key,
    #     subgraph_id=subgraph_id,
    #     pool_addresses=[main_pool_address],
    #     start_date=start_date,
    #     end_date=end_date,
    #     hdf5_path=hdf5_path,
    #     min_rows=min_rows,
    #     mode=hdf5_mode
    # )
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }
    # --- Prepare test dataset ---
    print("Preparing test dataset...")
    test_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[main_pool_address],
        features=features,
        target=target,
        n_lags=args.n_lags,
        split='test',
        split_dates=split_dates,
        verbose=1
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if len(test_dataset) == 0:
        print("No test data available.")
        return
    # --- Load model ---
    print("Loading model...")
    input_size = len(features) + 1 # +1 for the target variable
    if args.model_type == "lstm":
        model = ZeroInflatedLSTM(
            input_size=input_size,
            n_lags=args.n_lags,
            lstm_units=args.lstm_units,
            dense_units=args.dense_units
        )
    elif args.model_type == "transformer":
        model = ZeroInflatedTransformer(
            input_size=input_size,
            n_lags=args.n_lags,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dense_units=args.dense_units,
            dropout=args.dropout
        )
    else:
        print(f"Unknown model_type: {args.model_type}")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # --- Model finetuning ---
    print("Finetuning model on test pool on training+validation dates...")
    finetune_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[main_pool_address],
        features=features,
        target=target,
        n_lags=args.n_lags,
        split='train',
        split_dates={'train_start': split_dates['train_start'], 'train_end': split_dates['val_end']},
        verbose=1
    )
    finetune_val_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[main_pool_address],
        features=features,
        target=target,
        n_lags=args.n_lags,
        split='val',
        split_dates={'train_start': split_dates['train_start'], 'train_end': split_dates['val_end'], 'val_start': split_dates['val_start'], 'val_end': split_dates['val_end']},
        verbose=1
    )
    finetune_loader = DataLoader(finetune_dataset, batch_size=args.finetune_batch_size, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=args.finetune_batch_size, shuffle=False)
    if len(finetune_dataset) == 0:
        print("No data available for finetuning on this pool.")
    else:
        model.fit(
            finetune_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            verbose=1,
            val_loader=finetune_val_loader,
            early_stopping_patience=10
        )
        print("Finetuning complete.")
    # --- Model predictions ---
    X = test_dataset.X
    y_cls = test_dataset.y_cls
    y_reg = test_dataset.y_reg
    y_reg_pred, y_cls_pred = model.predict(X)
    # Print number of zero class predictions
    n_zero_pred = np.sum(y_cls_pred == 1)
    n_total = len(y_cls_pred)
    print(f"Zero class predictions: {n_zero_pred} out of {n_total} ({n_zero_pred/n_total:.2%})")

    # --- Custom loss on test set ---
    print("Calculating custom loss on test set...")
    with torch.no_grad():
        X_scaled = model.scale_X(X)
        y_cls_tensor = y_cls.unsqueeze(1)
        y_reg_tensor = model.scale_y_reg(y_reg).unsqueeze(1)
        cls_pred, reg_pred = model(X_scaled)
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
    # Use model's scaler for fair comparison
    y_reg_tensor_naive_scaled = model.scale_y_reg(y_reg_tensor_naive).unsqueeze(1)
    y_reg_tensor_true_scaled = model.scale_y_reg(y_reg_tensor_true).unsqueeze(1)
    # Naive loss: use true y_cls, naive y_reg
    naive_loss = model.__class__.custom_zi_loss(y_cls_tensor_naive, y_reg_tensor_naive_scaled, y_cls_tensor_naive, y_reg_tensor_true_scaled).item()
    print(f"Naive baseline custom loss: {naive_loss:.8f}")
    # --- Plot ---
    print("Saving figures...")
    y_reg_pred[y_cls_pred==1] = 0  # Set predicted liquidity return to 0 where cls_pred is 1
    save_actual_vs_predicted(y_reg_np, y_reg_pred, title=f"Actual vs Predicted {target} (Test Set)", filename=f"{args.model_type}_actual_vs_predicted_{target}.png")
    save_actual_vs_predicted(y_reg_np[mask], naive_pred[mask], title=f"Naive: Actual vs Predicted {target} (Test Set)", filename=f"naive_actual_vs_predicted_{target}.png")

if __name__ == "__main__":
    main()