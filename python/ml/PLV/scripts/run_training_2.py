import os
import torch
from random import shuffle
import random
import argparse
import sys
from python.ml.PLV.data_io import load_pool_data, make_lps_dataset_from_pool_dict, get_saved_pool_addresses
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
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=False, default=None, help='Comma-separated list of features')
    parser.add_argument('--target', type=str, required=False, default=None, help='Target column name')
    args = parser.parse_args()

    print("[run_training.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # --- User configuration ---
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    model_dir = os.path.join("python/ml/PLV/models")
    model_path = os.path.join(model_dir, f"{args.model_name}.pt")

    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        features = [
            'price_return', 'price_volatility_3h', 'price_volatility_6h', 'price_volatility_24h',
            'liquidity_volatility_3h', 'liquidity_volatility_6h', 'liquidity_volatility_24h',
            'price_ma_3h', 'price_ma_6h', 'price_ma_24h',
            'liquidity_ma_3h', 'liquidity_ma_6h', 'liquidity_ma_24h',
            'hour', 'day_of_week', 'month', 'season'
        ]
    if args.target is not None:
        target = args.target
    else:
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

    # --- LOPO Validation ---
    print("Selecting N pools and loading into memory...")
    # Get all available pool addresses from HDF5
    available_addresses = get_saved_pool_addresses(hdf5_path)
    random.seed(42)
    random.shuffle(available_addresses)
    selected_addresses = available_addresses[:args.n_pools]
    # Always include main_pool_address
    if args.main_pool_address not in selected_addresses:
        selected_addresses.append(args.main_pool_address)
    # Efficiently load only selected pools into memory
    selected_pools = {}
    for addr in selected_addresses:
        try:
            selected_pools[addr] = load_pool_data(hdf5_path, addr)
        except Exception as e:
            print(f"Warning: Could not load pool {addr}: {e}")
    print(f"Number of pools loaded: {len(selected_pools)}")
    for left_out in selected_addresses:
        print(f"\n=== LOPO Fold: Leaving out pool {left_out} ===")
        train_pools = [addr for addr in selected_addresses if addr != left_out]
        leftout_val_dataset = make_lps_dataset_from_pool_dict(
            selected_pools, [left_out], features, target, n_lags, 'val', split_dates, verbose=0)
        if len(leftout_val_dataset) == 0:
            print(f"Warning: Skipping LOPO fold for pool {left_out} due to zero validation samples.")
            continue
        train_dataset = make_lps_dataset_from_pool_dict(
            selected_pools, train_pools, features, target, n_lags, 'train', split_dates, verbose=0)
        print(f"Train samples: {len(train_dataset)}, Left-out Val samples: {len(leftout_val_dataset)}")
        input_size = len(features)+1
        model = ZeroInflatedLSTM(
            input_size=input_size,
            n_lags=n_lags,
            lstm_units=lstm_units,
            dense_units=dense_units
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_leftout_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X, y_cls, y_reg in train_loader:
                X_np = X.numpy()
                n_samples, n_lags_, n_features_ = X_np.shape
                X_reshaped = X_np.reshape(-1, n_features_)
                y_reg_np = y_reg.numpy().reshape(-1, 1)
                model.feature_scaler.partial_fit(X_reshaped)
                model.target_reg_scaler.partial_fit(y_reg_np)
                X_scaled = model.feature_scaler.transform(X_reshaped).reshape(n_samples, n_lags_, n_features_)
                y_reg_scaled = model.target_reg_scaler.transform(y_reg_np).flatten()
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
                y_cls = y_cls.to(device).unsqueeze(1)
                y_reg_tensor = torch.tensor(y_reg_scaled, dtype=torch.float32, device=device).unsqueeze(1)
                optimizer.zero_grad()
                cls_pred, reg_pred = model(X_tensor)
                loss = model.__class__.custom_zi_loss(cls_pred, reg_pred, y_cls, y_reg_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X.size(0)
            leftout_val_loss = model.evaluate(leftout_val_dataset)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_dataset):.6f}, Left-out Val Loss: {leftout_val_loss:.6f}")
            if leftout_val_loss < best_leftout_val_loss:
                best_leftout_val_loss = leftout_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}. Best Left-out Val Loss: {best_leftout_val_loss:.6f}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        # Save the best model for this fold
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
