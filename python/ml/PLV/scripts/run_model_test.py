import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python.ml.PLV.data_io import LPsDataset, fetch_and_save_pools
from python.ml.PLV.model import ZeroInflatedLSTM, custom_zi_loss

# Baseline naive prediction
def naive_predict(series):
    """Naive model: predicts next value as the current value (persistence)."""
    return series.shift(1)

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Liquidity Return"):
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
    plt.show()

def main():
    # --- Config ---
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    model_path = os.path.join("python/ml/PLV/models", "zero_inflated_lstm.pt")
    features = [
        'price_return', 'price_volatility_3h', 'price_volatility_6h', 'price_volatility_24h',
        'liquidity_volatility_3h', 'liquidity_volatility_6h', 'liquidity_volatility_24h',
        'price_ma_3h', 'price_ma_6h', 'price_ma_24h',
        'liquidity_ma_3h', 'liquidity_ma_6h', 'liquidity_ma_24h',
        'hour', 'day_of_week', 'month', 'season'
    ]
    target = 'liquidity_return'
    
    api_key = "d1762c97d76a973e078c5536742bd237" 
    subgraph_id = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"  
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    min_rows = 100
    pool_tier = None  # 'LOW', 'MEDIUM', 'HIGH', or None for all
    hdf5_mode = 'x'  # 'w' = overwrite, 'a' = append, 'x' = fail if exists
    main_pool_address = "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed" 
    print("Downloading test pool data (if not already)...")
    fetch_and_save_pools(
        api_key=api_key,
        subgraph_id=subgraph_id,
        pool_addresses=[main_pool_address],
        start_date=start_date,
        end_date=end_date,
        hdf5_path=hdf5_path,
        min_rows=min_rows,
        mode=hdf5_mode
    )

    n_lags = 6
    lstm_units = 16
    dense_units = 8
    batch_size = 128

    # --- Split dates for train/val/test ---
    split_dates = {
        'train_end': '2025-03-31',
        'val_end': '2025-05-01'
    }
    # --- Prepare test dataset ---
    print("Preparing test dataset...")
    test_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[main_pool_address],
        features=features,
        target=target,
        n_lags=n_lags,
        split='test',
        split_dates=split_dates,
        verbose=1
    )
    # with pd.HDFStore("python/ml/PLV/data/uniswap_pools_data.h5", "r") as store:
    #     df = store["pool_0xcbcdf9626bc03e24f779434178a73a0b4bad62ed"]
    #     print(df["datetime"].min(), df["datetime"].max())
    if len(test_dataset) == 0:
        print("No test data available.")
        return
    # --- Load model ---
    print("Loading model...")
    input_size = len(features)
    model = ZeroInflatedLSTM(
        input_size=input_size,
        n_lags=n_lags,
        lstm_units=lstm_units,
        dense_units=dense_units
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # --- Model predictions ---
    X = test_dataset.X
    y_cls = test_dataset.y_cls
    y_reg = test_dataset.y_reg
    y_reg_pred, y_cls_pred = model.predict(X)
    # --- Custom loss on test set ---
    print("Calculating custom loss on test set...")
    with torch.no_grad():
        X_scaled = model.scale_X(X)
        y_cls_tensor = y_cls.unsqueeze(1)
        y_reg_tensor = model.scale_y_reg(y_reg).unsqueeze(1)
        cls_pred, reg_pred = model(X_scaled)
        test_loss = custom_zi_loss(cls_pred, reg_pred, y_cls_tensor, y_reg_tensor).item()
    print(f"ZeroInflatedLSTM test custom loss: {test_loss:.8f}")
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
    naive_loss = custom_zi_loss(y_cls_tensor_naive, y_reg_tensor_naive_scaled, y_cls_tensor_naive, y_reg_tensor_true_scaled).item()
    print(f"Naive baseline custom loss: {naive_loss:.8f}")
    # --- Plot ---
    print("Plotting results...")
    plot_actual_vs_predicted(y_reg_np, y_reg_pred, title="LSTM: Actual vs Predicted Liquidity Return (Test Set)")
    plot_actual_vs_predicted(y_reg_np[mask], naive_pred[mask], title="Naive: Actual vs Predicted Liquidity Return (Test Set)")

if __name__ == "__main__":
    main()