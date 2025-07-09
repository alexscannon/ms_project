from typing import Optional
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

class GaussianCluster:
    """
    Class cluster object
    """

    def __init__(self, feature_dim: int, class_id: int, device: torch.device):
        self.feature_dim = feature_dim
        self.class_id = class_id
        self.device = device

        # Core cluster statistics
        self.mu = torch.zeros(feature_dim, device=self.device) # initialize a zeros vector
        self.sigma = torch.eye(n=feature_dim, device=self.device) # identity matrix
        self.num_of_exampels = 0

        # Cached computations
        self.inv_sigma = torch.eye(n=feature_dim, device=self.device)
        self.cholesky_decomposition = torch.eye(n=feature_dim, device=self.device)
        self.log_determinate = 0.0

    def initial_cluster_update(self, class_dataloader: DataLoader) -> None:
        """
        Updates the cluster's core statistics and cached computations based on training data.

        Args:
            class_dataloader (torch.utils.data.DataLoader): A batches' feature of shape (n_exmaples, feature_dim)
        Returns:
            None
        """

        pbar = tqdm(class_dataloader, desc="Initializing class clusters")
        for batch_idx, (inputs, targets)

            if features.shape[0] == 0:
                logging.error(f"Issue with cluster initialization. Not training examples provided...")
                return

            features_mu = features.mean(dim=0)
            self.mu += features
            self.num_of_exampels += inputs.shape[0]

        self.mu /= self.num_of_exampels


    def batch_update_cluster(self, features: torch.Tensor, alpha: int = 0.95) -> None:
        """
        Updates the cluster's core statistics and cached computations based on current batch.
        Difference to initialization is the exponential moving average component
        Args:
            features (torch.Tensor): A batches' feature of shape (n_exmaples, feature_dim)
        Returns:
            None
        """
        if features.shape[0] == 0:
            return

        features_mu = features.mean(dim=0)
        self.mu =