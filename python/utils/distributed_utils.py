import os
import functools
import sys

import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import state_dict as dist_state_dict

def setup():

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
    """Destroy the process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)


def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    # We do *not* want to write to the same location with multiple
    # processes at the same time.
    if is_root_process():
        torch.save(*args, **kwargs)


def save_full_model(model, optimizer=None, *args, **kwargs):
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

    save0(cpu_state, *args, **kwargs)
            

def load_full_model(model, optimizer=None, *args, **kwargs):
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


