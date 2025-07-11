import os
import torch
from python.ml.PLV.data_io import LPsDataset
from python.ml.PLV.model import ZeroInflatedLSTM


def main():
    # --- User configuration ---
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    model_dir = os.path.join("python/ml/PLV/models")
    model_path = os.path.join(model_dir, "zero_inflated_lstm.pt")

    features = [
        'price_return', 'price_volatility_3h', 'price_volatility_6h', 'price_volatility_24h',
        'liquidity_volatility_3h', 'liquidity_volatility_6h', 'liquidity_volatility_24h',
        'price_ma_3h', 'price_ma_6h', 'price_ma_24h',
        'liquidity_ma_3h', 'liquidity_ma_6h', 'liquidity_ma_24h',
        'hour', 'day_of_week', 'month', 'season'
    ]
    target = 'liquidity_return'  
    n_lags = 6
    batch_size = 64
    lstm_units = 16
    dense_units = 8
    lr = 0.001
    epochs = 20

    # --- Split dates for train/val/test ---
    split_dates = {
        'train_end': '2025-03-31',
        'val_end': '2025-05-01'
    }

    # --- Prepare dataset ---
    print("Preparing dataset...")
    train_dataset = LPsDataset(
        hdf5_path=hdf5_path,
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
