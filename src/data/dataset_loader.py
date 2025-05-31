from typing import Tuple
from omegaconf import DictConfig
import torch
from torchvision import datasets, transforms
import os
import logging
from torch.utils.data import Subset, DataLoader

from src.data.cifar100_dataset import CIFAR100Dataset
from src.data.utils import ClassRemappingDataset

DATASET_REGISTRY = {
    'cifar100': datasets.CIFAR100,
    'tiny_imagenet': datasets.ImageFolder,
}

def create_ood_detection_datasets(config: DictConfig, checkpoint_data: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create ID and OOD datasets based on class_info from the checkpoint object.
    The `pretrained_ind_dataloader` will have its labels remapped for fitting OOD detectors
    that expect consecutive class labels (0 to N-1).

    Args:
        config (DictConfig): Configuration
        checkpoint_data (dict): Checkpoint dictionary containing 'class_info'.
                                'class_info' is expected to follow the structure from
                                pre-training, including 'pretrain_classes', 'left_out_classes',
                                'pretrained_ind_indices', 'left_out_ind_indices', and 'class_mapping'.
    Returns:
        left_out_ind_dataloader (torch.utils.data.DataLoader): DataLoader for ID samples not used for OOD fitting (original labels).
        ood_dataloader (torch.utils.data.DataLoader): DataLoader for OOD samples (original labels).
        pretrained_ind_dataloader (torch.utils.data.DataLoader): DataLoader for ID samples used for OOD fitting (remapped labels).
    """
    class_info = checkpoint_data.get('class_info', None)
    if class_info is None:
        raise ValueError("'class_info' is missing from checkpoint_data. Cannot create OOD dataloaders.")

    if config.data.name == 'cifar100':
        dataset_wrapper = CIFAR100Dataset(config)
    elif config.data.name == 'tiny_imagenet':
        raise NotImplementedError("Tiny ImageNet is not supported yet.")
    else:
        raise ValueError(f"Dataset {config.data.name} not supported.")

    # Create Left-Out IND dataset (samples from pretrain classes)
    left_out_ind_indices = class_info.get('left_out_indices', None)
    if left_out_ind_indices is None:
        raise ValueError("'left_out_indices' is missing from class_info. Cannot create Left-Out IND dataloader.")

    class_mapping = class_info.get('class_mapping', None)
    if class_mapping is None:
        raise ValueError("'class_mapping' is missing from class_info. Cannot create Left-Out IND dataloader.")

    left_out_ind_subset = Subset(dataset_wrapper.train, left_out_ind_indices)
    left_out_ind_dataloader = DataLoader(
        dataset=ClassRemappingDataset(left_out_ind_subset, class_mapping),
        batch_size=config.data.batch_size,
        shuffle=True, # Shuffle for fitting
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # TODO: Temporary fix for the checkpoint data structure until new training run is conducted (#3 -> #4)
    if hasattr(checkpoint_data, 'pretrained_ind_indices'):
        pretrained_ind_indices = checkpoint_data['pretrained_ind_indices']
    else:
        # Get all the in-distribution class indicies
        all_ind_indices = list(range(len(dataset_wrapper.train)))
        # Of the pretrain classes, get the indices that are not the left out indices
        pretrained_ind_indices = [i for i in all_ind_indices if i not in left_out_ind_indices]

    # Create ID dataset (samples from pretrain classes)
    pretrained_ind_subset = Subset(dataset_wrapper.train, pretrained_ind_indices)
    pretrained_ind_dataloader = DataLoader(
        dataset=ClassRemappingDataset(pretrained_ind_subset, class_mapping),
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # Create OOD dataset (samples from 'left_out_classes')
    # TODO: After another training run is conducted, change the key to 'left_out_classes'
    left_out_classes = class_info.get('continual_classes', None)
    if left_out_classes is None:
        raise ValueError("'continual_classes' is missing from class_info. Cannot create OOD dataloader.")
    logging.info(f"Found {len(left_out_classes)} left out classes")
    logging.info(f"Left out classes: {left_out_classes}")
    logging.info(f"Training set targets: {dataset_wrapper.train.targets}")
    ood_class_label_set = set(left_out_classes)
    try:
        # Efficient way if targets attribute exists (like in torchvision CIFAR datasets)
        ood_sample_indices = [i for i, label in enumerate(dataset_wrapper.train.targets) if label in ood_class_label_set]
    except AttributeError:
        # Slower fallback if .targets is not directly available
        logging.warning("dataset_wrapper.train.targets not found, iterating to find OOD samples. This might be slow.")
        ood_sample_indices = [i for i, (_, label) in enumerate(dataset_wrapper.train) if label in ood_class_label_set]
    logging.info(f"Found {len(ood_sample_indices)} OOD samples")
    ood_subset = Subset(dataset_wrapper.train, ood_sample_indices)
    ood_dataloader = DataLoader(
        dataset=ood_subset, # No label remapping needed for OOD dataset because it is not used for fitting the OOD detector
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    logging.info(f"Created ID dataset with {len(pretrained_ind_dataloader)} samples from {len(pretrained_ind_indices)} classes")
    logging.info(f"Created Left-Out ID dataset with {len(left_out_ind_dataloader)} samples from {len(left_out_ind_indices)} classes")
    logging.info(f"Created OOD dataset with {len(ood_dataloader)} samples from {len(left_out_classes)} classes")

    return left_out_ind_dataloader, ood_dataloader, pretrained_ind_dataloader