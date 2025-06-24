import os
import random
import shutil
from typing import Any, Tuple
from omegaconf import DictConfig
import torch
from torchvision import datasets, transforms
import logging
from torch.utils.data import Subset, DataLoader, Dataset
import numpy as np
from colorama import Fore, Style

from src.data.cifar100_dataset import CIFAR100Dataset
from src.data.tiny_imagenet_dataset import TinyImagenetDataset
from src.data.utils import ClassRemappingDataset


# def datainfo(config: DictConfig) -> dict[str, Any]:
#     if config.data.name == 'cifar10':
#         print(Fore.YELLOW + '*' * 80)
#         logging.debug('CIFAR10')
#         print('*'*80 + Style.RESET_ALL)
#         n_classes = config.data.num_classes
#         img_mean, img_std = config.data.mean, config.data.std
#         img_size = config.data.image_size

#     elif config.data.name == 'cifar100':
#         print(Fore.YELLOW+'*'*80)
#         logging.debug('CIFAR100')
#         print('*'*80 + Style.RESET_ALL)
#         n_classes = config.data.num_classes
#         img_mean, img_std = config.data.mean, config.data.std
#         img_size = config.data.image_size

#     elif config.data.name == 'svhn':
#         print(Fore.YELLOW+'*'*80)
#         logging.debug('SVHN')
#         print('*' * 80 + Style.RESET_ALL)
#         n_classes = config.data.num_classes
#         img_mean, img_std = config.data.mean, config.data.std
#         img_size = config.data.image_size

#     elif config.data.name == 'tiny_imagenet':
#         print(Fore.YELLOW + '*' * 80)
#         logging.debug('T-IMNET')
#         print('*' * 80 + Style.RESET_ALL)
#         n_classes = config.data.num_classes
#         img_mean, img_std = config.data.mean, config.data.std
#         img_size = config.data.image_size

#     data_info = {
#         'n_classes': n_classes,
#         'stat': (img_mean, img_std),
#         'img_size': img_size
#     }

#     return data_info


def dataload(config: DictConfig, checkpoint_data: dict):
    """
    TODO: Documentation
    """

    if config.data.name == 'cifar10':
        raise NotImplementedError("[Error] CIFAR10 dataset is not supported...")

    elif config.data.name == 'svhn':
        raise NotImplementedError("[Error] SVHN dataset is not supported...")

    elif config.data.name == 'cifar100':
        cifar100DatasetWrapper = CIFAR100Dataset(config)
        train_dataset = cifar100DatasetWrapper.train
        val_dataset = cifar100DatasetWrapper.val
        logging.info(f"Length of raw train dataset: {len(train_dataset)}")
        logging.info(f"Length of raw val dataset: {len(val_dataset)}")

        left_in_ind_dataset, left_out_ind_dataset, ood_dataset = create_sub_datasets(checkpoint_data, train_dataset, val_dataset)

    elif config.data.name == 'tiny_imagenet':
        tinyImagenetDatasetWrapper = TinyImagenetDataset(config)
        left_in_ind_dataset = tinyImagenetDatasetWrapper.train_ind_in_dataset
        left_out_ind_dataset = tinyImagenetDatasetWrapper.train_ind_out_dataset
        ood_dataset = tinyImagenetDatasetWrapper.ood_dataset

    else:
        raise NotImplementedError(f"[Error] {config.data.name} not supported...")

    left_in_ind_dataloader = DataLoader(
        dataset=left_in_ind_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    left_out_ind_dataloader = DataLoader(
        dataset=left_out_ind_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    ood_dataloader = DataLoader(
        dataset=ood_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    return left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader




def create_sub_datasets(checkpoint_data: dict, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ID and OOD datasets based on class_info from the checkpoint object if supported by dataset (ex., CIFAR100).
    The `pretrained_ind_dataloader` will have its labels remapped for fitting OOD detectors
    that expect consecutive class labels (0 to N-1).

    Args:
        config (DictConfig): Configuration
        checkpoint_data (dict): Checkpoint dictionary containing 'class_info'.
                                'class_info' is expected to follow the structure from
                                pre-training, including 'pretrain_classes', 'left_out_classes',
                                'pretrained_ind_indices', 'left_out_ind_indices', and 'ind_class_mapping'.
    Returns:
        left_in_ind_dataset
        left_out_ind_dataset
        ood_dataset
    """
    class_info = checkpoint_data.get('class_info', None)
    if class_info is None:
        logging.error("'class_info' is missing from checkpoint_data. Cannot create OOD dataloaders.")
        raise ValueError("[Error] 'class_info' is missing from checkpoint_data. Cannot create OOD dataloaders.")

    pretrained_ind_indices = class_info.get('pretrained_ind_indices', [])
    left_out_ind_indices = class_info.get('left_out_ind_indices', [])
    ood_example_idxs = class_info.get('ood_example_idxs', [])
    pretrain_class_mapping = class_info.get('pretrain_class_mapping', [])

    left_in_ind_dataset = ClassRemappingDataset(
        dataset=Subset(train_dataset, pretrained_ind_indices),
        class_mapping=pretrain_class_mapping
    )

    left_out_ind_dataset = ClassRemappingDataset(
        dataset=Subset(train_dataset, left_out_ind_indices),
        class_mapping=pretrain_class_mapping
    )

    ood_dataset = Subset(train_dataset, ood_example_idxs)

    logging.info(f"Length of left_in_ind_dataset: {len(left_in_ind_dataset)}")
    logging.info(f"Length of left_out_ind_dataset: {len(left_out_ind_dataset)}")
    logging.info(f"Length of left_out_ind_dataset: {len(left_out_ind_dataset)}")

    return left_in_ind_dataset, left_out_ind_dataset, ood_dataset

