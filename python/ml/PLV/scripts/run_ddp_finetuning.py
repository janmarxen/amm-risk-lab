"""
run_ddp_finetuning.py

Distributed finetuning script for Uniswap V3 ML models using PyTorch DDP.

High-level steps:
1. Initialize distributed process group and set up device for each rank.
2. Parse command-line arguments for model/data configuration and finetuning pool.
3. Load pretrained model from checkpoint.
4. Fit feature and target scalers on the finetuning pool's training data (on rank 0)
   and broadcast to all ranks. These scalers are saved to enable consistent scaling
   during the testing phase.
5. Construct training and validation datasets for the finetuning pool using the
   fitted scalers to ensure consistent scaling across train/val/test splits.
6. Wrap the model with DistributedDataParallel and finetune on the specified pool.
7. Save the finetuned model and scalers (only on rank 0).
8. Clean up and destroy the process group.

Note: Unlike pretraining where per-pool scaling is used, finetuning fits a single
scaler per feature/target that is shared across train/val/test for consistency.
"""
import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import json
from python.ml.PLV.data_io import LPsDataset, fit_scalers
from python.ml.PLV.model import ZeroInflatedTransformer
from python.utils.distributed_utils import save_model_arch0, setup, print0, save0, destroy_process_group, load_full_model, save_scalers0, load_scalers


def main(args):
    local_rank, rank, device = setup()

    print0("[run_ddp_finetuning.py] Configuration:")
    for k, v in vars(args).items():
        print0(f"  {k}: {v}")
    sys.stdout.flush()

    finetune_pool_address = args.finetune_pool_address
    pretrained_model_name = args.pretrained_model_name
    finetuned_model_name = args.finetuned_model_name
    hdf5_path = args.hdf5_path
    model_path = os.path.join("python/ml/PLV/models", f"{pretrained_model_name}.pt")
    finetuned_model_path = os.path.join("python/ml/PLV/models", f"{finetuned_model_name}.pt")

    # Load model architecture from JSON
    arch_path = os.path.splitext(model_path)[0] + '_arch.json'
    with open(arch_path, 'r') as f:
        arch = json.load(f)
    features = arch['features']
    targets = arch['targets']  # Multi-task targets
    
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
    input_size = len(features) + len(targets)  # features + target lags (2 targets)
    model = ZeroInflatedTransformer(
        input_size=input_size,
        n_lags=arch['n_lags'],
        d_model=arch['d_model'],
        num_heads=arch['num_heads'],
        num_layers=arch['num_layers'],
        dense_units=arch['dense_units'],
        dropout=arch['dropout']
    )
    # Load full model checkpoint
    model, _ = load_full_model(model, None, model_path, map_location=device)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # Fit scalers on rank 0
    print0("Fitting scalers on rank 0...")
    if rank == 0:
        feature_scaler, target_reg_scalers = fit_scalers(
            hdf5_path=hdf5_path,
            pool_addresses=[finetune_pool_address],
            features=features,
            targets=targets,
            split_dates=split_dates,
            verbose=0,
            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 4)),
            sample_size_pct=1.0, # Use full dataset since its small
        )
    else:
        feature_scaler = StandardScaler()
        target_reg_scalers = [StandardScaler(), StandardScaler()]
    # Broadcast fitted scalers from rank 0 to all ranks
    scaler_list = [feature_scaler, target_reg_scalers]
    dist.broadcast_object_list(scaler_list, src=0)
    feature_scaler, target_reg_scalers = scaler_list
    print0("Scalers loaded and broadcasted to all ranks.")
    # --- Model finetuning ---
    print0("Finetuning model on test pool on training+validation dates...")
    finetune_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        targets=targets,
        n_lags=arch['n_lags'],
        split='train',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers,
        verbose=1
    )
    finetune_val_dataset = LPsDataset(
        hdf5_path=hdf5_path,
        pool_addresses=[finetune_pool_address],
        features=features,
        targets=targets,
        n_lags=arch['n_lags'],
        split='val',
        split_dates=split_dates,
        feature_scaler=feature_scaler,
        target_reg_scalers=target_reg_scalers,
        verbose=1
    )
    finetune_loader = DataLoader(finetune_dataset, batch_size=args.finetune_batch_size, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=args.finetune_batch_size, shuffle=False)
    if len(finetune_dataset) == 0:
        print0("No data available for finetuning on this pool.")
    else:
        print0("Starting finetuning...")
        # Finetune model
        print0("Finetuning model...")
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
        save_model_arch0(
            finetuned_model_path,
            **arch
        )
        save_scalers0(feature_scaler, target_reg_scalers, os.path.splitext(finetuned_model_path)[0] + '_scalers.pkl')
        print0("Finetuning complete.")
        destroy_process_group()

def parse_args():
        parser = argparse.ArgumentParser()
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
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--pretrained_model_name', type=str, required=True, help='Name of the pretrained model file')
        parser.add_argument('--finetuned_model_name', type=str, required=True, help='Name of the finetuned model file')
        parser.add_argument('--hdf5_path', type=str, required=True, help='Path to the HDF5 data file')
        return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    main(args)
