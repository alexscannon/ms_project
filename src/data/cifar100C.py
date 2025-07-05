import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from omegaconf import DictConfig

from src.data.numpy_dataset import NumpyDataset

class CIFAR100CDataset():
    def __init__(self, config: DictConfig) -> None:
        """ Initializes the CIFAR-100-C dataset loader.
        Args:
            config (DictConfig): Configuration object containing dataset parameters.
        Returns:
            None
        """
        self.config = config
        self.base_dir = config.continual_learning.cifar100C_location
        self.corruption_data_path = os.path.join(self.base_dir, config.continual_learning.corruption_type)
        self.imgs_per_severity = 10000

        self.full_dataset_inputs = np.load(f"{self.corruption_data_path}.npy")
        self.full_dataset_targets = np.load(os.path.join(self.base_dir, "labels.npy"))

        self.transform = self._get_transform()

    def get_dataset(self, severity_level: int) -> Dataset:
        """
        Returns a CIFAR-100-C dataset for a specific severity level.
        Args:
            severity_level (int): The severity level of the corruption (1 to 5).
        Returns:
            Dataset: A PyTorch Dataset object containing the CIFAR-100-C data for the specified severity level.
        """
        dataset_start_idx = (severity_level - 1) * self.imgs_per_severity
        dataset_end_idx = (dataset_start_idx) + self.imgs_per_severity

        inputs = self.full_dataset_inputs[dataset_start_idx: dataset_end_idx]
        targets = self.full_dataset_targets[dataset_start_idx: dataset_end_idx]

        return NumpyDataset(inputs, targets, self.transform)


    def _get_transform(self) -> transforms.Compose:
        """
        Returns a torchvision transform for CIFAR-100-C dataset.
        The transform includes resizing, converting to tensor, and normalization.
        Args:
            None
        Returns:
            transforms.Compose: A composed transform for CIFAR-100-C dataset.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.continual_learning.cifar100C_mean,
                std=self.config.continual_learning.cifar100C_std
            )
        ])

