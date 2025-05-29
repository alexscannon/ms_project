import os
from omegaconf import DictConfig
import torch
import logging

def get_checkpoint_dict(dataset_name: str, config: DictConfig, device: str) -> dict:
    """
    Get the checkpoint dictionary for the given dataset.
    Args:
        dataset_name (str): Name of the dataset
        config (DictConfig): Configuration object
        device (str): Device to load the checkpoint on
    Returns:
        checkpoint_data (dict): Checkpoint dictionary
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state_dict, # Model weights
            'optimizer_state_dict': optimizer_state_dict, # Optimizer state
            'history': history, # Training history
            'class_info': {
                'num_of_classes': num_of_classes, # Old name for this property "n_classes"
                'pretrain_classes': pretrain_class_indicies,
                'left_out_classes': ood_class_indicies, # Old name for this property "continual_classes"
                'left_out_ind_indices': pretrained_left_out_indices, # Old name for this property "left_out_indices"
                'class_mapping': class_mapping
            }
        }
    """
    # Only CIFAR-100 has a separate checkpoint directory
    logging.info(f"Loading checkpoint for {dataset_name} from {config.model.backbone.location}")
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

