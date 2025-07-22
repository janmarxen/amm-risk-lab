import os
import torch
from random import shuffle
import random
import argparse
import sys
from python.ml.PLV.data_io import LPsDataset, get_saved_pool_addresses
from python.ml.PLV.model import ZeroInflatedLSTM, ZeroInflatedTransformer


def main():
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
    parser.add_argument('--main_pool_address', type=str, required=True)
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=False, default=None, help='Comma-separated list of features')
    parser.add_argument('--target', type=str, required=False, default=None, help='Target column name')
    args = parser.parse_args()
    # Check required arguments for LSTM
    if args.model_type == "lstm" and args.lstm_units is None:
        parser.error("--lstm_units is required when --model_type is 'lstm'")

    print("[run_training.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # --- User configuration ---
    # hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    model_dir = os.path.join("python/ml/PLV/models")
    model_path = os.path.join(model_dir, f"{args.model_name}.pt")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print("[run_training.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.target is not None:
        target = args.target
    else:
        print("[run_training.py] ERROR: --target argument must be specified.")
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

    # --- Prepare dataset ---
    print("Preparing dataset...")
    pool_addresses = get_saved_pool_addresses(hdf5_path)
    print(f"Number of pools in HDF5: {len(pool_addresses)}")
    # Shuffle and limit the number of pools for training (set seed for reproducibility)
    random.seed(42)
    shuffle(pool_addresses)
    N = args.n_pools # Limit to first N pools for training
    pool_addresses = pool_addresses[:N]
    # Always include main_pool_address
    if args.main_pool_address not in pool_addresses:
        pool_addresses.append(args.main_pool_address)
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
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    # --- Train model ---
    print("Training model with early stopping...")
    input_size = len(features)+1 # +1 for the target variable
    if args.model_type == "lstm":
        model = ZeroInflatedLSTM(
            input_size=input_size,
            n_lags=n_lags,
            lstm_units=lstm_units,
            dense_units=dense_units
        )
    elif args.model_type == "transformer":
        model = ZeroInflatedTransformer(
            input_size=input_size,
            n_lags=n_lags,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dense_units=dense_units,
            dropout=args.dropout
        )
    else:
        print(f"Unknown model_type: {args.model_type}")
        sys.exit(1)
    model.fit(
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=1,
        val_dataset=val_dataset,
        early_stopping_patience=20
    )
    print("Model training complete.")
    # --- Save model ---
    print("Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
