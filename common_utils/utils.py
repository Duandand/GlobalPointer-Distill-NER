import logging
import os
import random

import numpy as np
import torch


def save_pytorch_model(output_dir, model, file_name, max_save, type):
    """Save a trained model.

    Args:
        output_dir: a directory where the model will be saved
        model: the model or optimizer object to save
        file_name: string, the name of model or optimizer
        max_save: int, the maximum number of model stores
        type: string, either 'model' or 'optimizer'
    """
    logging.info(f"** ** * Saving {type}: {file_name} ** ** * ")
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    file_ckpt = os.path.join(output_dir, file_name)
    torch.save(
        model_to_save.state_dict(), file_ckpt, _use_new_zipfile_serialization=False
    )
    logging.info(f"** ** * Successfully saved {type}: {file_name} ** ** * ")


def restore_model(model, model_path, device, map_location):
    logging.info(f"Load from {model_path}!!!!!")
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict, strict=False)  # todo
    model.to(device)
    return model


def device_config(local_rank: int, no_cuda: bool):
    """Get device and n_gpu for training.

    Args:
        local_rank (int): If used multi-gpu.
        no_cuda (bool): Whether cuda is available.

    Returns:
        _type_: devce and n_gpu
    """
    if local_rank == -1 or no_cuda:
        logging.info(f"no_cuda {no_cuda}")
        if torch.cuda.is_available() and not no_cuda:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available() and not no_cuda:
            # Apple Silicon GPU (MPS) fallback when CUDA is unavailable
            device = torch.device("mps")
            n_gpu = 1
        else:
            device = torch.device("cpu")
            n_gpu = 0
        logging.info("device %s n_gpu %d", device, n_gpu)
    else:
        torch.distributed.init_process_group(backend="nccl")
        # local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1

        logging.info(
            "rank %d device %s n_gpu %d distributed training %r",
            torch.distributed.get_rank(),
            device,
            n_gpu,
            bool(local_rank != -1),
        )
    return device, n_gpu


def random_seed_config(seed: int, n_gpu: int):
    """Setting Random seed.

    Args:
        seed (int): Ensure random seeds are consistent!!!!!
        n_gpu (int): If used multi-gpu.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"setting seed {seed} success!!!!!")
