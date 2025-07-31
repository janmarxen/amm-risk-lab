import os
import torch
from torch.utils.data import DataLoader
import argparse
import sys
from python.ml.PLV.data_io import LPsDataset
from python.ml.PLV.model import ZeroInflatedLSTM, ZeroInflatedTransformer
from python.utils.distributed_utils import setup, print0, save0, destroy_process_group, load_full_model
from python.utils.data_utils import load_scalers

def main():
    local_rank, rank, device = setup()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lags', type=int, required=True)
    parser.add_argument('--lstm_units', type=int, required=False, help='Number of LSTM units (required for LSTM)')
    parser.add_argument('--dense_units', type=int, required=True)
    parser.add_argument('--train_start', type=str, required=True)
    parser.add_argument('--train_end', type=str, required=True)
    parser.add_argument('--val_start', type=str, required=True)
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--test_start', type=str, required=True)
    parser.add_argument('--test_end', type=str, required=True)
    parser.add_argument('--finetune_pool_address', type=str, required=True)
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pretrained_model_name', type=str, required=True, help='Name of the pretrained model file')
    parser.add_argument('--finetuned_model_name', type=str, required=True, help='Name of the finetuned model file')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # Check required arguments for LSTM
    if args.model_type == "lstm" and args.lstm_units is None:
        parser.error("--lstm_units is required when --model_type is 'lstm'")
    print0("[run_ddp_testing.py] Configuration:")
    for k, v in vars(args).items():
        print0(f"  {k}: {v}")
    sys.stdout.flush()
    finetune_pool_address = args.finetune_pool_address
    pretrained_model_name = args.pretrained_model_name
    finetuned_model_name = args.finetuned_model_name
    hdf5_path = os.path.join("/p/scratch/training2529", "uniswap_pools_data.h5")
    model_path = os.path.join("python/ml/PLV/models", f"{pretrained_model_name}.pt")
    finetuned_model_path = os.path.join("python/ml/PLV/models", f"{finetuned_model_name}.pt")
    if args.features is not None:
        features = [f.strip() for f in args.features.split(',')]
    else:
        print0("[run_ddp_testing.py] ERROR: --features argument must be specified.")
        sys.exit(1)
    if args.target is not None:
        target = args.target
    else:
        print0("[run_ddp_testing.py] ERROR: --target argument must be specified.")
        sys.exit(1)
    split_dates = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
        'test_start': args.test_start,
        'test_end': args.test_end
    }
    # --- Load model ---
    print0("Loading model...")
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
        print0(f"Unknown model_type: {args.model_type}")
        destroy_process_group()
        sys.exit(1)
    # Load full model checkpoint
    model, _ = load_full_model(model, None, model_path, map_location=device)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # Load scalers
    scaler_path = os.path.splitext(model_path)[0] + '_scalers.pkl'
    feature_scaler, target_reg_scaler = load_scalers(scaler_path)
    # --- Model finetuning ---
    print0("Finetuning model on test pool on training+validation dates...")
    finetune_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        target=target,
        n_lags=args.n_lags,
        split='train',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scaler=target_reg_scaler,
        verbose=1
    )
    finetune_val_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        target=target,
        n_lags=args.n_lags,
        split='val',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scaler=target_reg_scaler,
        verbose=1
    )
    finetune_loader = DataLoader(finetune_dataset, batch_size=args.finetune_batch_size, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=args.finetune_batch_size, shuffle=False)
    if len(finetune_dataset) == 0:
        print0("No data available for finetuning on this pool.")
    else:
        model.module.fit_distributed(
            train_loader=finetune_loader,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            verbose=1 if rank == 0 else 0,
            val_loader=finetune_val_loader,
            early_stopping_patience=10,
            device=device
        )
        save0(model, finetuned_model_path)
        print0("Finetuning complete.")
        destroy_process_group()

if __name__ == "__main__":
    main()
