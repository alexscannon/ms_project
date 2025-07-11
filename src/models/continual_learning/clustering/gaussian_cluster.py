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

        # Required computations for running calculations
        self.num_of_exampels = 0
        self.x_xt = torch.zeros(feature_dim, feature_dim, device=self.device)

        # Core cluster statistics
        self.mu = torch.zeros(feature_dim, device=self.device) # initialize a zeros vector
        self.sigma = torch.eye(n=feature_dim, device=self.device) # identity matrix

        # Cached computations
        self.inv_sigma = torch.eye(n=feature_dim, device=self.device)
        self.cholesky_decomposition = torch.eye(n=feature_dim, device=self.device)
        self.log_determinate = 0.0

    def initial_cluster_update(self, features: torch.Tensor) -> None:
        """
        Updates the cluster's core statistics and cached computations based on training data.

        Args:
            features (torch.Tensor): A batches' feature of shape (n_exmaples, feature_dim)
        Returns:
            None
        """


        features_total_sum = features.sum(dim=0)
        old_total_sum = self.mu * self.num_of_exampels
        self.num_of_exampels += features.shape[0] # Increase the total number of elements by the batch count
        self.mu = (features_total_sum + old_total_sum) / self.num_of_exampels

        features_x_xT = features.T @ features
        self.x_xt += features_x_xT
        e_xxt = self.x_xt / self.num_of_exampels # E[x*x^T]

        outer_mu = torch.outer(self.mu, self.mu)
        self.sigma = e_xxt - outer_mu
        self.sigma += torch.eye(self.feature_dim, device=self.device) * 1e-4

        # Update cached computations
        self.inv_sigma = torch.inverse(self.sigma)
        self.cholesky_decomposition = torch.linalg.cholesky(self.sigma)
        self.log_determinate = torch.slogdet(self.sigma)



    # def batch_update_cluster(self, features: torch.Tensor, alpha: int = 0.95) -> None:
    #     """
    #     Updates the cluster's core statistics and cached computations based on current batch.
    #     Difference to initialization is the exponential moving average component
    #     Args:
    #         features (torch.Tensor): A batches' feature of shape (n_exmaples, feature_dim)
    #     Returns:
    #         None
    #     """
    #     if features.shape[0] == 0:
    #         return

    #     features_mu = features.mean(dim=0)
    #     self.mu =