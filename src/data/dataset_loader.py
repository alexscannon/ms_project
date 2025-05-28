from typing import Dict, Tuple
from omegaconf import DictConfig
import torch
from torchvision import datasets, transforms
import os
import logging
from torch.utils.data import Subset

from src.data.cifar100_dataset import CIFAR100Dataset

DATASET_REGISTRY = {
    'cifar100': datasets.CIFAR100,
    'tiny_imagenet': datasets.ImageFolder,
}

def create_transform(config):
    """Create a transform pipeline based on config."""
    transform_list = []

    # Add resize if image_size is specified
    if hasattr(config.dataset, 'image_size'):
        transform_list.append(
            transforms.Resize(config.dataset.image_size)
        )

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Add normalization if mean and std are specified
    if hasattr(config.dataset, 'mean') and hasattr(config.dataset, 'std'):
        transform_list.append(
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std)
        )

    return transforms.Compose(transform_list)

def load_dataset(config):
    """Load dataset based on configuration and wrap in ContinualDataset."""
    dataset_name = config.dataset.name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not supported. "
                       f"Available datasets: {list(DATASET_REGISTRY.keys())}")

    # Create transform
    transform = create_transform(config)

    # Get dataset class
    dataset_class = DATASET_REGISTRY[dataset_name]

    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == 'imagenet':
        # ImageNet requires separate train and val directories
        train_dir = os.path.join(config.dataset.root, 'train')
        val_dir = os.path.join(config.dataset.root, 'val')

        if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
            raise ValueError(
                f"ImageNet directories not found at {config.dataset.root}. "
                "Expected structure: \n"
                "root/\n"
                "  train/\n"
                "    n01440764/\n"
                "      *.JPEG\n"
                "  val/\n"
                "    n01440764/\n"
                "      *.JPEG"
            )

        train_dataset = dataset_class(
            root=config.dataset.root,
            split='train',
            transform=transform
        )

        test_dataset = dataset_class(
            root=config.dataset.root,
            split='val',
            transform=transform
        )
    else:
        # Load train and test datasets using the root path from config
        train_dataset = dataset_class(
            root=config.dataset.root,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = dataset_class(
            root=config.dataset.root,
            train=False,
            download=True,
            transform=transform
        )

    # Wrap in ContinualDataset
    # Could move back to main.py if we want to separate the continual dataset wrapper from the dataset loading.
    continual_dataset = ContinualDataset(
        train_dataset,
        test_dataset,
        num_classes=config.dataset.num_classes
    )

    return continual_dataset

def create_ood_detection_datasets(config: DictConfig, checkpoint_data: Dict) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create ID and OOD datasets based on class_info from the checkpoint object.

    Args:
        config: Configuration
        checkpoint_data: Class information from the checkpoint
    Returns:
        Tuple of (id_dataset, ood_dataset)
    """
    # TODO: Change property names to match the ones in the notebook if another training run is conducted (#3 -> #4)
    class_info = checkpoint_data['class_info']

    if config.data.name == 'cifar100':
        dataset = CIFAR100Dataset(config)
    elif config.data.name == 'tiny_imagenet':
        raise NotImplementedError("Tiny ImageNet is not supported yet.")
    else:
        raise ValueError(f"Dataset {config.data.name} not supported.")

    # Create ID dataset (samples from pretrain classes)
    left_out_ind_indices = class_info['left_out_indices']
    left_out_ind_dataset = Subset(dataset.train, left_out_ind_indices)

    # Create OOD dataset (samples from continual/OOD classes)
    left_out_classes = dict(class_info['continual_classes'])
    ood_indices = [i for i, (_, label) in enumerate(dataset.train) if label in left_out_classes]
    ood_dataset = Subset(dataset.train, ood_indices)

    logging.info(f"Created ID dataset with {len(left_out_ind_dataset)} samples from {len(left_out_ind_indices)} classes")
    logging.info(f"Created OOD dataset with {len(ood_dataset)} samples from {len(class_info['continual_classes'])} classes")

    return left_out_ind_dataset, ood_dataset