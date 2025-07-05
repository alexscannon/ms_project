from typing import Tuple
from omegaconf import DictConfig
import logging
from torch.utils.data import Subset, DataLoader, Dataset

from .cifar100 import CIFAR100Dataset
from .cifar100C import CIFAR100CDataset
from .tiny_imagenet import TinyImagenetDataset
from .tiny_imagenet_c import TinyImagenetCDataset
from src.data.utils import ClassRemappingDataset


def dataload(config: DictConfig, checkpoint_data: dict) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader | None]:
    """
    Load the in-distribution and out-of-distribution datasets based on the configuration.
    Args:
        config (DictConfig): Configuration object containing dataset parameters.
        checkpoint_data (dict): Checkpoint dictionary containing 'class_info'
                                'class_info' is expected to follow the structure from
                                pre-training, including 'pretrain_classes', 'left_out_classes',
                                'pretrained_ind_indices', 'left_out_ind_indices', and 'ind_class_mapping'.
    Returns:
        left_in_ind_dataloader (torch.utils.data.DataLoader): Dataloader for in-distribution examples used during pre-training.
        left_out_ind_dataloader (torch.utils.data.DataLoader): Dataloader for in-distribution examples not used during pre-training.
        ood_dataloader (torch.utils.data.DataLoader): Dataloader for out-of-distribution examples.
        corruption_dataloader (torch.utils.data.DataLoader | None): Dataloader for the corruption dataset used for continual learning scenario.
    Raises:
        NotImplementedError: If the dataset is not supported.
    """
    corruption_dataloader = None

    if config.data.name == 'cifar10':
        raise NotImplementedError("[Error] CIFAR10 dataset is not supported...")

    elif config.data.name == 'svhn':
        raise NotImplementedError("[Error] SVHN dataset is not supported...")

    elif config.data.name == 'cifar100':
        cifar100DatasetWrapper = CIFAR100Dataset(config)
        if config.continual_learning.corruption_injection:
            corruption_dataset = CIFAR100CDataset(config=config).get_dataset(config.continual_learning.corruption_severity)

        train_dataset = cifar100DatasetWrapper.train
        val_dataset = cifar100DatasetWrapper.val # Not currently used but could be useful in the future.

        left_in_ind_dataset, left_out_ind_dataset, ood_dataset = create_sub_datasets(checkpoint_data, train_dataset)

    elif config.data.name == 'tiny_imagenet':
        tinyImagenetDatasetWrapper = TinyImagenetDataset(config)

        left_in_ind_dataset = tinyImagenetDatasetWrapper.train_ind_in_dataset
        left_out_ind_dataset = tinyImagenetDatasetWrapper.train_ind_out_dataset
        ood_dataset = tinyImagenetDatasetWrapper.ood_dataset
        if config.continual_learning.corruption_injection:
            corruption_dataset = TinyImagenetCDataset(config=config).get_dataset(config.continual_learning.corruption_severity)

    else:
        raise NotImplementedError(f"[Error] {config.data.name} not supported...")


    # Create DataLoaders for the datasets
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

    if config.continual_learning.corruption_injection:
        corruption_dataloader = DataLoader(
            dataset=corruption_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )

    return left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader, corruption_dataloader




def create_sub_datasets(checkpoint_data: dict, train_dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ID and OOD datasets based on class_info from the checkpoint object if supported by dataset (ex., CIFAR100).
    The `pretrained_ind_dataloader` will have its labels remapped for fitting OOD detectors
    that expect consecutive class labels (0 to N-1).

    Args:
        checkpoint_data (dict): Checkpoint dictionary containing 'class_info'.
                                'class_info' is expected to follow the structure from
                                pre-training, including 'pretrain_classes', 'left_out_classes',
                                'pretrained_ind_indices', 'left_out_ind_indices', and 'ind_class_mapping'.
        train_dataset (torch.utils.data.Dataset): Complete training dataset.
    Returns:
        left_in_ind_dataset (torch.utils.data.Dataset): Dataset containing in-distribution examples that were used during pre-training.
        left_out_ind_dataset (torch.utils.data.Dataset): Dataset containing in-distribution examples that were not used during pre-training.
        ood_dataset (torch.utils.data.Dataset): Dataset containing out-of-distribution examples.
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

