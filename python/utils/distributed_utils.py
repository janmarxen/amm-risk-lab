"""
distributed_utils.py

Utilities for distributed training and checkpointing in PyTorch, including process group setup, root process detection, atomic printing, and model saving/loading.
"""
import os
import functools
import sys
import pickle
import json

import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import state_dict as dist_state_dict

def setup():
    """
    Initialize the distributed process group and set up the CUDA device for this process.
    Returns:
        local_rank (int): Local rank of the process on the node.
        rank (int): Global rank of the process in the world.
        device (torch.device): CUDA device assigned to this process.
    """

    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')

    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))

    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])

    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)
    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)

    # Different random seed for each process.
    torch.random.manual_seed(1000 + torch.distributed.get_rank())

    return local_rank, rank, device

def destroy_process_group():
    """
    Destroy the distributed process group if initialized.
    """
    """Destroy the process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

functools.lru_cache(maxsize=None)
def is_root_process():
    """
    Return True if this process is the root process (rank 0), else False.
    Returns:
        bool: True if root process, False otherwise.
    """

    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def print0(*args, **kwargs):
    """
    Print only on the root process (rank 0).
    Args:
        *args: Arguments to print.
        **kwargs: Keyword arguments to print.
    """

    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)


def save0(*args, **kwargs):
    """
    Save only the underlying model's state_dict (not the DDP wrapper) to the given path, only on the root process.
    This ensures the model can be loaded outside a distributed context.
    Usage:
        save0(model, path)  # model can be DDP-wrapped or not
    Args:
        model: PyTorch model (possibly DDP-wrapped)
        path: Path to save the state_dict
    """
    if is_root_process():
        model, path = args[0], args[1]
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)


def save_full_model(model, optimizer=None, *args, **kwargs):
    """
    Gather all model parameters to rank 0 on CPU and save the model (and optimizer) state dict using torch.save, only on the root process.
    Args:
        model: PyTorch model to save.
        optimizer: (Optional) PyTorch optimizer to save.
        *args: Arguments for torch.save.
        **kwargs: Keyword arguments for torch.save.
    """

    """Stream all model parameters to rank 0 on the CPU, then pass all
    other given arguments to `torch.save` to save the model, but only on
    the root process.
    """
    state_dict_options = dist_state_dict.StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    cpu_state_dict = dist_state_dict.get_model_state_dict(
        model,
        options=state_dict_options,
    )
    cpu_state = {'model': cpu_state_dict}

    if optimizer is not None:
        optim_state_dict = dist_state_dict.get_optimizer_state_dict(
            model,
            optimizer,
            options=state_dict_options,
        )
        cpu_state['optimizer'] = optim_state_dict

    # Save the cpu_state dict directly, not via save0 (which expects a model)
    if is_root_process():
        torch.save(cpu_state, args[0])
            

def load_full_model(model, optimizer=None, *args, **kwargs):
    """
    Load model and optimizer state dict from a checkpoint file using torch.load.
    Args:
        model: PyTorch model to load into.
        optimizer: (Optional) PyTorch optimizer to load into.
        *args: Arguments for torch.load.
        **kwargs: Keyword arguments for torch.load.
    Returns:
        tuple: (model, optimizer) with loaded state dicts.
    """

    """Pass all other given arguments to `torch.load` and load the
    resulting state dict into the given model.
    """

    state_dict = torch.load(*args, **kwargs)

    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
        

    model.load_state_dict(state_dict['model'])
    return model, optimizer


def atomic_print(*args, device=None, **kwargs):
    """
    Print from only one process at a time, in rank order. Optionally include device info.
    Args:
        *args: Arguments to print.
        device: (Optional) Device info to include in prefix.
        **kwargs: Keyword arguments to print.
    """

    """
    Print from only one process at a time, in rank order. Optionally include device info.
    """
    if not torch.distributed.is_initialized():
        print(*args, **kwargs)
        return
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    for r in range(world_size):
        torch.distributed.barrier()
        if r == rank:
            prefix = f"[rank {rank} | device {device}] " if device is not None else f"[rank {rank}] "
            print(prefix, *args, **kwargs)
            sys.stdout.flush()
    torch.distributed.barrier()


def save_scalers0(feature_scaler, target_reg_scaler, path):
    """
    Save feature and target scalers to a file using pickle, only on rank 0.
    Args:
        feature_scaler: Fitted feature scaler (e.g., StandardScaler)
        target_reg_scaler: Fitted target scaler (e.g., StandardScaler)
        path: Path to save the scalers (should end with .pkl)
    """
    if is_root_process():
        with open(path, 'wb') as f:
            pickle.dump({'feature_scaler': feature_scaler, 'target_reg_scaler': target_reg_scaler}, f)

def load_scalers(path):
    """
    Load feature and target scalers from a pickle file.
    Args:
        path: Path to the saved scalers (.pkl)
    Returns:
        (feature_scaler, target_reg_scaler) 
    """
    with open(path, 'rb') as f:
        scalers = pickle.load(f)
        return scalers['feature_scaler'], scalers['target_reg_scaler']
    
def save_model_arch0(model_path, n_lags, d_model, num_heads, num_layers, dense_units, dropout, features, target):
    """
    Save transformer model architecture as JSON, only on rank 0.
    Args:
        model_path (str): Path to the model .pt file (used as base for JSON filename)
        n_lags, d_model, num_heads, num_layers, dense_units, dropout: Transformer hyperparameters
        features (list): List of feature names
        target (str): Target column name
    """
    arch_path = os.path.splitext(model_path)[0] + '_arch.json'
    if is_root_process():
        arch_dict = {
            'n_lags': n_lags,
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dense_units': dense_units,
            'dropout': dropout,
            'features': features,
            'target': target
        }
        with open(arch_path, 'w') as f:
            json.dump(arch_dict, f, indent=2)

def save_gridsearch_result(result, output_file):
    """
    Atomically append a single grid search result to the results file with file locking.
    Args:
        result (dict): Grid search result containing params and val_loss
        output_file (str): Path to the output JSONL file
    """
    import fcntl
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a') as f:
        # Lock the file to prevent race conditions
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        # Write the result as a single JSON line
        json.dump(result, f)
        f.write('\n')  # Newline separator
        
        # Unlock happens automatically when file closes