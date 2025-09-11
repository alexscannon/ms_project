
from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections.abc import Sized
from torch.utils.data import Dataset
import logging


class ClassRemappingDataset(Dataset):
    """
    A dataset wrapper that remaps class labels given the ViT was trained on a subset of the classes.
    """
    def __init__(self, dataset: Dataset, class_mapping: dict):
        """
        Args:
            dataset (torch.utils.data.Dataset): The original dataset (e.g., a Subset).
            class_mapping (dict): A dictionary mapping original class labels
                                  to new, consecutive class labels (e.g., {orig_label: new_label}).
        """
        self.dataset = dataset
        self.class_mapping = class_mapping

    def __getitem__(self, index: int) -> tuple:
        img, original_target = self.dataset[index]
        if original_target not in self.class_mapping:
            # This error indicates a mismatch between the data subset and the provided mapping.
            # The subset should only contain samples whose original labels are keys in class_mapping.
            logging.info(f"class mapping: {self.class_mapping}")
            raise ValueError(
                f"Original target {original_target} not found in class_mapping. \n"
                f"Ensure the input dataset to ClassRemappingDataset only contains samples \n"
                f"from the classes defined in the class_mapping."
            )
        # Remap the original target to the new target (e.g., 23 -> 0, 11 -> 1, 93 -> 2, ...)
        # New target is in the range [0, self.expected_num_classes - 1]
        new_target = self.class_mapping[original_target]
        return img, new_target

    def __len__(self) -> int:
        assert isinstance(self.dataset, Sized)
        return len(self.dataset)


def get_embeddings(dataloader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts all feature embeddings for a given dataset using the provided encoder model.

    Args:
        dataloader (DataLoader): DataLoader providing batches of input data. Directly passing in dataloader to avoid
                                    class creation errors.
    Returns:
        all_embeddings (np.ndarray): A numpy array containing the extracted embeddings for the entire dataset.

    """
    model.to(device)
    model.eval()

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Generating Embeddings.."):
            images = images.to(device)
            output = model(images)
            embeddings = output.cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)