import os
import torch
from torch.utils.data import DataLoader
from data_io import LPsDataset
from model import ZeroInflatedLSTM
from sklearn.model_selection import ParameterGrid
import numpy as np


def grid_search_lstm(
    hdf5_path,
    features,
    target,
    split_dates,
    param_grid,
    epochs=10,
    batch_size=128,
    verbose=1
):
    """
    Perform grid search for ZeroInflatedLSTM hyperparameters using LPsDataset and DataLoader.

    Args:
        hdf5_path: Path to HDF5 file.
        features: List of feature columns to use.
        target: Name of target column.
        split_dates: Dict with keys 'train_end', 'val_end' for splitting.
        param_grid: dict of hyperparameters (lstm_units, dense_units, batch_size, n_lags, lr, etc.)
        epochs: Number of training epochs.
        batch_size: Default batch size if not in param_grid.
        verbose: Print progress if True.
    Returns:
        best_model: Trained model with best validation RMSE.
        best_params: Dict of best hyperparameters.
        results: List of (params, val_rmse) tuples.
    """
    best_score = float('inf')
    best_params = None
    best_model = None
    results = []
    for params in ParameterGrid(param_grid):
        n_lags = params.get('n_lags', 12)
        batch_size_ = params.get('batch_size', batch_size)
        # Create datasets for this n_lags
        train_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            features=features,
            target=target,
            n_lags=n_lags,
            split='train',
            split_dates=split_dates
        )
        val_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            features=features,
            target=target,
            n_lags=n_lags,
            split='val',
            split_dates=split_dates
        )
        input_size = len(features)
        model = ZeroInflatedLSTM(
            input_size=input_size,
            n_lags=n_lags,
            lstm_units=params.get('lstm_units', 32),
            dense_units=params.get('dense_units', 16)
        )
        # --- Training ---
        model.fit(train_dataset, epochs=epochs, batch_size=batch_size_, lr=params.get('lr', 0.001), verbose=verbose)
        # --- Validation ---
        rmse = model.evaluate(val_dataset)
        results.append((params, rmse))
        if verbose:
            print(f"Params: {params}, Val RMSE: {rmse:.8f}")
        if rmse < best_score:
            best_score = rmse
            best_params = params
            best_model = model
    if verbose:
        print(f"\nBest params: {best_params}, Best Val RMSE: {best_score:.8f}")
    return best_model, best_params, results


def main():
    # --- User configuration ---
    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    features = [
        'price_return', 'price_volatility_3h', 'price_volatility_6h', 'price_volatility_24h',
        'liquidity_volatility_3h', 'liquidity_volatility_6h', 'liquidity_volatility_24h',
        'price_ma_3h', 'price_ma_6h', 'price_ma_24h',
        'liquidity_ma_3h', 'liquidity_ma_6h', 'liquidity_ma_24h',
        'hour', 'day_of_week', 'month', 'season'
    ]
    target = 'price_return'  # Example target, adjust as needed

    batch_size = 128

    # --- Split dates for train/val/test ---
    split_dates = {
        'train_end': '2025-03-31',
        'val_end': '2025-05-01'
    }

    # --- Hyperparameter grid ---
    param_grid = {
        'lstm_units': [16, 32],
        'dense_units': [8, 16],
        'n_lags': [6, 12],
        'batch_size': [64, 128],
        'lr': [0.001, 0.0005]
    }

    # --- Run grid search ---
    best_model, best_params, results = grid_search_lstm(
        hdf5_path=hdf5_path,
        features=features,
        target=target,
        split_dates=split_dates,
        param_grid=param_grid,
        epochs=10,
        batch_size=batch_size,
        verbose=1
    )

    print("\nBest hyperparameters:", best_params)
    print("Grid search results:")
    for params, val_loss in results:
        print(f"Params: {params}, Val Loss: {val_loss:.8f}")

    # --- Save model ---
    model_dir = os.path.join("python/ml/PLV/models")
    model_path = os.path.join(model_dir, "best_zero_inflated_lstm.pt")
    torch.save(best_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

