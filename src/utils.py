import os
from omegaconf import DictConfig
import torch
import logging

def get_checkpoint_dict(dataset_name: str, config: DictConfig, device: str) -> dict:
    # Only CIFAR-100 has a separate checkpoint directory
    if dataset_name == 'cifar100':
        checkpoint_path = os.path.join(config.model.backbone.location, dataset_name, config.model.backbone.model_filename)
    else:
        checkpoint_path = os.path.join(config.model.backbone.location, config.model.backbone.model_filename)

    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # --- Load Checkpoint ---
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device) # Load directly to target device if possible
        logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    return checkpoint_data

