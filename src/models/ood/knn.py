from typing import Callable
from omegaconf import DictConfig
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
from typing import Tuple
from tqdm import tqdm


class KNNDetector:
    def __init__(self, model: torch.nn.Module, config: DictConfig):
        """
        KNN-based OOD detector.

        This method stores feature representations of in-distribution data
        and uses distance to k-th nearest neighbor as OOD score.

        Args:
            model (torch.nn.Module): The model to use for feature extraction
            config (DictConfig): Configuration for the KNN detector
        """
        self.model = model
        self.config = config
        self.k = config.knn.k
        self.metric = config.knn.metric
        self.threshold = config.knn.threshold if hasattr(config.knn, 'threshold') else 0.0

        self.knn_index = None
        self.is_fitted = False

    def fit(self, dataloader: torch.utils.data.DataLoader, feature_extractor_fn: Callable, num_classes: int) -> None:
        """
        Fit the KNN detector by storing features from in-distribution data.

        Args:
            dataloader (torch.utils.data.DataLoader): In-distribution training data
            feature_extractor_fn (callable): Function to extract features from inputs
            num_classes (int): Number of classes (not used but kept for consistency)
        """
        logging.info(f"Fitting KNN detector with k={self.k} and metric={self.metric}...")

        all_features = []
        device = next(self.model.parameters()).device  # Get the device of the model

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Extracting features for KNN fitting...", unit="batch")
            for batch_idx, batch in enumerate(pbar):
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None

                # Move data to the same device as the model
                x = x.to(device)

                # Extract features using the provided function
                features, _ = feature_extractor_fn(x, self.model)

                # Move to CPU and convert to numpy for sklearn
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()

                all_features.append(features)


        # Concatenate all features
        self.training_features = np.concatenate(all_features, axis=0)
        logging.info(f"KNN detector fitted with {self.training_features.shape[0]} training samples, "
                    f"feature dimension: {self.training_features.shape[1]}")

        # Initialize and fit the KNN index
        self.knn_index = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.metric,
            algorithm='auto'
        )
        self.knn_index.fit(self.training_features)

        self.is_fitted = True
        logging.info("KNN detector fitting completed.")

    def get_knn_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute KNN-based OOD scores.

        Args:
            features (torch.Tensor): Features of test samples

        Returns:
            torch.Tensor: KNN scores (negative distances to k-th neighbor)
        """
        if not self.is_fitted:
            raise RuntimeError("KNN detector must be fitted before computing scores")

        # Convert to numpy if needed
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().detach().numpy()
            return_tensor = True
        else:
            features_np = features
            return_tensor = False

        # Compute distances to k nearest neighbors
        distances, _ = self.knn_index.kneighbors(features_np)

        # Use distance to k-th nearest neighbor as score
        # We use the k-th neighbor (index k-1) since kneighbors returns k neighbors
        kth_distances = distances[:, -1]  # Last column contains k-th neighbor distance

        # Return negative distances as scores (higher score = more ID-like)
        # This makes it consistent with other detectors where higher score = more ID
        scores = -kth_distances

        # Always return a torch.Tensor to match the return type annotation
        return torch.tensor(scores, dtype=torch.float32)


    def predict_ood_knn(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts OOD status based on Mahalanobis scores.

        Args:
            features (torch.Tensor): Input features of shape (B, D).

        Returns:
            mahalanobis_scores (torch.Tensor): The computed scores or negative distances.
            is_ood (torch.Tensor): Boolean tensor indicating OOD status, True if OOD.
        """
        knn_scores = self.get_knn_scores(features)
        # Higher score (less negative/more positive) means more in-distribution.
        is_ood = knn_scores < self.threshold
        return knn_scores, is_ood
