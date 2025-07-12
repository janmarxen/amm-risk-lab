import os
import torch
from random import shuffle
import argparse
import sys
from python.ml.PLV.data_io import LPsDataset, get_saved_pool_addresses
from python.ml.PLV.model import ZeroInflatedLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lags', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lstm_units', type=int, required=True)
    parser.add_argument('--dense_units', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=False, default="zero_inflated_lstm.pt")
    parser.add_argument('--main_pool_address', type=str, required=True)
    args = parser.parse_args()

    print("[run_training.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # --- User configuration ---
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    model_dir = os.path.join("python/ml/PLV/models")
    model_path = os.path.join(model_dir, f"{args.model_name}.pt")

    features = [
        'price_return', 'price_volatility_3h', 'price_volatility_6h', 'price_volatility_24h',
        'liquidity_return', 'liquidity_volatility_3h', 'liquidity_volatility_6h', 'liquidity_volatility_24h',
        'price_ma_3h', 'price_ma_6h', 'price_ma_24h',
        'liquidity_ma_3h', 'liquidity_ma_6h', 'liquidity_ma_24h',
        'hour', 'day_of_week', 'month', 'season'
    ]
    target = 'liquidity_return'  
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
    # Shuffle and limit the number of pools for training
    shuffle(pool_addresses)
    N = 1000 # Limit to first N pools for training
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
    # --- Train model ---
    print("Training model...")
    input_size = len(features)
    model = ZeroInflatedLSTM(
        input_size=input_size,
        n_lags=n_lags,
        lstm_units=lstm_units,
        dense_units=dense_units
    )
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size, lr=lr, verbose=1)    
    print("Model training complete.")
    # --- Save model ---
    print("Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
