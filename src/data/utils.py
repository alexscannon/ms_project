import os
import traceback
from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections.abc import Sized
from torch.utils.data import Dataset
import logging
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig


logger = logging.getLogger("msproject")

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
            logger.info(f"class mapping: {self.class_mapping}")
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

    all_embeddings_np = np.concatenate(all_embeddings, axis=0)
    all_labels_np= np.concatenate(all_labels, axis=0)

    # Standarize features
    logger.info(f"Standardizing embeddings (zero mu, unit sigma)...")
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings_np)

    return all_embeddings_scaled, all_labels_np

def load_embeddings(config: DictConfig, model: nn.Module, device: torch.device, full_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns saved (normalized) embeddings & labels or computes, saves, then returns
    new (normalized) embeddings & labels.

    Args:
        config
        model
        device
        full_dataloader
    Returns:

    """
    # Location of embeddings file path (.pth format)
    embeddings_file_path = os.path.join(config.embeddings_location, config.data.embeddings_filename)
    embeddings, true_labels = None, None

    if os.path.exists(embeddings_file_path):
        try:
            logger.info(f"Existing computed and normalized embeddedings...")
            data_dict = torch.load(embeddings_file_path)
            # Convert tensors back to np.ndarray for scikit-learn compatability
            embeddings = data_dict.get('embeddings', torch.empty(0)).numpy()
            true_labels = data_dict.get('true_labels', torch.empty(0)).numpy()
        except Exception as e:
            logger.info(f"[ERROR] Failed to load saved embedding data. ({type(e).__name__}: {e})")
            traceback.print_exc()
    else:
        try:
            logger.info(f"No existing embeddedings found...")
            embeddings, true_labels = get_embeddings(dataloader=full_dataloader, model=model, device=device)
            logger.info(f"Saving normalized embeddings at path: {embeddings_file_path}")
            data = {
                "embeddings": torch.from_numpy(embeddings),
                "true_labels": torch.from_numpy(true_labels),
            }
            torch.save(data, embeddings_file_path)
            logger.info("Successfully created and saved embeddings data...")
        except Exception as e:
            logger.info(f"[ERROR] Failed to create or save embedding data. ({type(e).__name__}: {e})")
            traceback.print_exc()

    return embeddings, true_labels