import os
import torch
from torch.utils.data import DataLoader
import argparse
import sys
import itertools
import random
from random import shuffle

from sklearn.model_selection import ParameterGrid
from python.ml.PLV.data_io import LPsDataset, get_saved_pool_addresses
from python.ml.PLV.model import ZeroInflatedLSTM, ZeroInflatedTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lags_list', type=str, required=True, help='Comma-separated list of n_lags')
    parser.add_argument('--batch_size_list', type=str, required=True, help='Comma-separated list of batch sizes')
    parser.add_argument('--lstm_units_list', type=str, required=False, help='Comma-separated list of LSTM units (LSTM only)')
    parser.add_argument('--d_model_list', type=str, required=False, help='Comma-separated list of d_model (Transformer only)')
    parser.add_argument('--num_heads_list', type=str, required=False, help='Comma-separated list of num_heads (Transformer only)')
    parser.add_argument('--num_layers_list', type=str, required=False, help='Comma-separated list of num_layers (Transformer only)')
    parser.add_argument('--dense_units_list', type=str, required=True, help='Comma-separated list of dense units')
    parser.add_argument('--dropout_list', type=str, required=False, help='Comma-separated list of dropout rates (Transformer only)')
    parser.add_argument('--lr_list', type=str, required=True, help='Comma-separated list of learning rates')
    parser.add_argument('--epochs_list', type=str, required=True, help='Comma-separated list of epochs')
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=False, default="zero_inflated_lstm_gs")
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'], help='Model type to use')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--main_pool_address', type=str, required=True)
    parser.add_argument('--n_pools', type=int, required=False, default=1000)
    parser.add_argument('--features', type=str, required=True, help='Comma-separated list of features')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    args = parser.parse_args()

    print("[run_gridsearch.py] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    hdf5_path = os.path.join("python/ml/PLV/data", "uniswap_pools_data.h5")
    model_dir = os.path.join("python/ml/PLV/models")
    features = [f.strip() for f in args.features.split(',')]
    target = args.target
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }
    pool_addresses = get_saved_pool_addresses(hdf5_path)
    print(f"Number of pools in HDF5: {len(pool_addresses)}")
    random.seed(42)
    shuffle(pool_addresses)
    N = args.n_pools
    pool_addresses = pool_addresses[:N]
    if args.main_pool_address not in pool_addresses:
        pool_addresses.append(args.main_pool_address)

    n_lags_grid = [int(x) for x in args.n_lags_list.split(',')]
    batch_size_grid = [int(x) for x in args.batch_size_list.split(',')]
    dense_units_grid = [int(x) for x in args.dense_units_list.split(',')]
    lr_grid = [float(x) for x in args.lr_list.split(',')]
    epochs_grid = [int(x) for x in args.epochs_list.split(',')]

    # --- Unified grid search for both model types ---
    if args.model_type == "lstm":
        model_class = ZeroInflatedLSTM
        param_grid = list(itertools.product(n_lags_grid, batch_size_grid, [int(x) for x in args.lstm_units_list.split(',')] if args.lstm_units_list else [32], dense_units_grid, lr_grid, epochs_grid))
        param_names = ["n_lags", "batch_size", "lstm_units", "dense_units", "lr", "epochs"]
    elif args.model_type == "transformer":
        model_class = ZeroInflatedTransformer
        d_model_grid = [int(x) for x in args.d_model_list.split(',')] if args.d_model_list else [32]
        num_heads_grid = [int(x) for x in args.num_heads_list.split(',')] if args.num_heads_list else [2]
        num_layers_grid = [int(x) for x in args.num_layers_list.split(',')] if args.num_layers_list else [2]
        dropout_grid = [float(x) for x in args.dropout_list.split(',')] if args.dropout_list else [0.1]
        param_grid = list(itertools.product(n_lags_grid, batch_size_grid, d_model_grid, num_heads_grid, num_layers_grid, dense_units_grid, dropout_grid, lr_grid, epochs_grid))
        param_names = ["n_lags", "batch_size", "d_model", "num_heads", "num_layers", "dense_units", "dropout", "lr", "epochs"]
    else:
        print(f"Unknown model_type: {args.model_type}")
        sys.exit(1)

    print(f"Grid search over {len(param_grid)} combinations.")
    best_loss = float('inf')
    best_model_state = None
    best_model_path = None
    best_params = None
    for i, params in enumerate(param_grid, 1):
        param_dict = dict(zip(param_names, params))
        print(f"\n[GridSearch] Combination {i}/{len(param_grid)}: " + ", ".join(f"{k}={v}" for k, v in param_dict.items()))
        n_lags = param_dict["n_lags"]
        batch_size = param_dict["batch_size"]
        dense_units = param_dict["dense_units"]
        lr = param_dict["lr"]
        epochs = param_dict["epochs"]
        # Model-specific params
        model_kwargs = dict(input_size=len(features)+1, n_lags=n_lags, dense_units=dense_units)
        if args.model_type == "lstm":
            model_kwargs["lstm_units"] = param_dict["lstm_units"]
        else:
            model_kwargs["d_model"] = param_dict["d_model"]
            model_kwargs["num_heads"] = param_dict["num_heads"]
            model_kwargs["num_layers"] = param_dict["num_layers"]
            model_kwargs["dropout"] = param_dict["dropout"]
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
        model = model_class(**model_kwargs)
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
        finetune_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=[args.main_pool_address],
            features=features,
            target=target,
            n_lags=n_lags,
            split='train',
            split_dates={'train_start': split_dates['train_start'], 'train_end': split_dates['val_end']}
        )
        finetune_val_dataset = LPsDataset(
            hdf5_path=hdf5_path,
            pool_addresses=[args.main_pool_address],
            features=features,
            target=target,
            n_lags=n_lags,
            split='val',
            split_dates={'val_start': split_dates['val_start'], 'val_end': split_dates['val_end']}
        )
        if len(finetune_dataset) == 0 or len(finetune_val_dataset) == 0:
            print("Skipping finetuning for this grid point: no data for main pool.")
            continue
        model.fit(
            finetune_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=1,
            val_dataset=finetune_val_dataset,
            early_stopping_patience=10
        )
        print("Finetuning complete.")
        val_loss = model.evaluate(finetune_val_dataset)
        print(f"Custom loss on main pool validation: {val_loss:.8f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_params = params
            best_model_path = os.path.join(model_dir, f"{args.model_name}_BEST.pt")
            print("New best model found!")
    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved to {best_model_path}")
        print("Best params: " + ", ".join(f"{k}={v}" for k, v in zip(param_names, best_params)))
        print(f"Best custom loss: {best_loss:.8f}")
    else:
        print("No valid model was trained and finetuned.")

if __name__ == "__main__":
    main()


