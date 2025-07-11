import torch
from torch.nn import Module
from torch import device as TorchDevice
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.utils import extract_features_and_logits
from src.models.continual_learning.clustering.gaussian_cluster import GaussianCluster

class StreamingGaussianClusters:
    def __init__(self, num_classes: int, training_dataloader: DataLoader, feature_dim: int, device: TorchDevice, model: Module):
        self.num_classes = num_classes
        self.training_dataloader = training_dataloader
        self.feature_dim = feature_dim
        self.device = device
        self.model = model
        self.clusters = { i: GaussianCluster(feature_dim=self.feature_dim, class_id=i, device=self.device) for i in range(self.num_classes) }

        self._construct_all_class_clusters()

    def _construct_all_class_clusters(self) -> None:
        """
        Constructs Gaussian clusters for all classes using the training data.
        Optimized for performance by accumulating statistics across all batches
        and performing expensive matrix operations only once per class.
        """
        # Initialize accumulators for each class
        class_counts = {i: 0 for i in range(self.num_classes)}
        class_sums = {i: torch.zeros(self.feature_dim, device=self.device) for i in range(self.num_classes)}
        class_x_xt = {i: torch.zeros(self.feature_dim, self.feature_dim, device=self.device) for i in range(self.num_classes)}

        # Process all batches and accumulate statistics
        pbar = tqdm(self.training_dataloader, desc="Accumulating statistics...")
        for batch_id, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Extract features
            with torch.no_grad():
                features, _ = extract_features_and_logits(x=inputs, model=self.model)

            # Update statistics for each class present in the batch
            unique_classes = targets.unique()
            for class_id in unique_classes:
                class_id_item = class_id.item()
                class_mask = targets == class_id
                class_features = features[class_mask]

                # Accumulate statistics
                class_counts[class_id_item] += class_features.shape[0]
                class_sums[class_id_item] += class_features.sum(dim=0)
                class_x_xt[class_id_item] += class_features.T @ class_features

        # Update each cluster once with accumulated statistics
        for class_id in range(self.num_classes):
            if class_counts[class_id] > 0:
                # Calculate mean
                mu = class_sums[class_id] / class_counts[class_id]

                # Update cluster
                cluster = self.clusters[class_id]
                cluster.num_of_exampels = class_counts[class_id]
                cluster.mu = mu
                cluster.x_xt = class_x_xt[class_id]

                # Calculate covariance and update all cluster statistics at once
                e_xxt = class_x_xt[class_id] / class_counts[class_id]
                outer_mu = torch.outer(mu, mu)
                cluster.sigma = e_xxt - outer_mu + torch.eye(self.feature_dim, device=self.device) * 1e-4

                # Compute cached values only once per class
                cluster.inv_sigma = torch.inverse(cluster.sigma)
                cluster.cholesky_decomposition = torch.linalg.cholesky(cluster.sigma)
                cluster.log_determinate = torch.slogdet(cluster.sigma)[1].item()


