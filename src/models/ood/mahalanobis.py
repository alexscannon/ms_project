import torch
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import logging

class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD detector.
    This detector needs to be fitted on in-distribution training data before use.
    Scores are negative Mahalanobis distances (higher is more in-distribution).
    """

    def __init__(self, model: torch.nn.Module, ood_config: DictConfig, device: str = 'cuda'):
        self.model = model # Used to infer feature dimensionality if needed, but fit takes feature_extractor
        self.ood_config = ood_config
        self.device = device
        self.threshold = ood_config.mahalanobis.threshold if hasattr(ood_config, 'mahalanobis') and hasattr(ood_config.mahalanobis, 'threshold') else 0.0 # Default threshold

        self.class_means = None
        self.precision_matrix = None
        self.fitted_num_classes = 0
        self.expected_num_classes = 0 # Set during fit

        # Regularization factor for covariance matrix inversion
        self.covariance_reg = ood_config.mahalanobis.covariance_reg if hasattr(ood_config, 'mahalanobis') and hasattr(ood_config.mahalanobis, 'covariance_reg') else 1e-5


    @property
    def is_fitted(self) -> bool:
        return self.class_means is not None and self.precision_matrix is not None

    def fit(self, train_dataloader: torch.utils.data.DataLoader, feature_extractor_fn: callable, num_classes: int):
        """
        Fit the Mahalanobis detector by computing class means and shared precision matrix.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for in-distribution training data.
            feature_extractor_fn (callable): Function that takes raw input x and returns (features, logits).
            num_classes (int): The total number of expected classes.
        """
        logging.info("Fitting Mahalanobis detector...")
        self.expected_num_classes = num_classes
        all_features_list = [[] for _ in range(self.expected_num_classes)]

        # Temporarily set model to eval if it's part of feature_extractor_fn's closure implicitly
        # self.model.eval() # feature_extractor_fn is expected to handle model's mode

        # Collect features for each class
        first_batch_features = None
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Mahalanobis Fit: Extracting Features"):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                features, _ = feature_extractor_fn(x) # Expects (features, logits)
                if first_batch_features is None:
                    first_batch_features = features

                for i in range(self.expected_num_classes):
                    class_mask = (y == i)
                    if class_mask.any():
                        all_features_list[i].append(features[class_mask].cpu())

        if first_batch_features is None:
            raise ValueError("Cannot fit Mahalanobis detector: training dataloader is empty or feature_extractor_fn did not yield features.")

        class_features = [torch.cat(fl, dim=0) if fl else torch.empty(0, first_batch_features.shape[-1]) for fl in all_features_list]

        # Compute class means only for classes with samples
        active_class_indices = [i for i, cf in enumerate(class_features) if cf.shape[0] > 0]
        if not active_class_indices:
            raise ValueError("No class has any samples in the training data for Mahalanobis fitting.")

        self.class_means = torch.stack([class_features[i].mean(dim=0) for i in active_class_indices]).to(self.device)
        self.fitted_num_classes = self.class_means.shape[0]
        logging.info(f"Mahalanobis: Computed means for {self.fitted_num_classes}/{self.expected_num_classes} classes.")
        if self.fitted_num_classes < self.expected_num_classes:
            logging.warning(f"Mahalanobis: Only {self.fitted_num_classes} classes had samples. Scores will be based on these classes.")

        # Compute shared covariance and precision matrix
        valid_features = torch.cat([class_features[i] for i in active_class_indices], dim=0).to(self.device)

        if valid_features.shape[0] == 0:
            raise ValueError("No data found to compute covariance matrix for Mahalanobis detector.")

        if valid_features.shape[0] < valid_features.shape[1]:
            logging.warning(f"Mahalanobis: Number of samples ({valid_features.shape[0]}) is less than feature dimensionality ({valid_features.shape[1]}). Using regularized covariance.")

        cov = torch.cov(valid_features.T)
        reg_cov = cov + torch.eye(cov.shape[0], device=self.device) * self.covariance_reg
        try:
            self.precision_matrix = torch.linalg.inv(reg_cov).to(self.device)
        except torch.linalg.LinAlgError:
            logging.warning("Mahalanobis: Covariance matrix inversion failed. Using pseudo-inverse.")
            self.precision_matrix = torch.linalg.pinv(reg_cov).to(self.device)

        logging.info("Mahalanobis detector fitted successfully.")


    def get_mahalanobis_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis scores for the given features.
        Scores are negative Mahalanobis distances (higher = more in-distribution).

        Args:
            features (torch.Tensor): Input features of shape (B, D).

        Returns:
            torch.Tensor: Mahalanobis scores of shape (B,).
        """
        if not self.is_fitted:
            raise RuntimeError("Mahalanobis detector has not been fitted. Call fit() first.")

        features = features.to(self.device)
        batch_size = features.shape[0]

        # Scores tensor will hold distances to each fitted class mean
        mahalanobis_distances_to_classes = torch.zeros(batch_size, self.fitted_num_classes, device=self.device)

        for i in range(self.fitted_num_classes):
            mean = self.class_means[i].reshape(1, -1) # Shape [1, D]
            diff = features - mean # Shape [B, D]

            # (x-mu)^T * Sigma^-1 * (x-mu)
            term1 = torch.matmul(diff, self.precision_matrix) # [B, D] @ [D, D] = [B, D]
            mahalanobis_distances_to_classes[:, i] = torch.sum(term1 * diff, dim=1) # Element-wise product and sum over D

        # OOD score is based on the minimum distance to any class mean
        # Smaller distance = more in-distribution
        min_distances, _ = torch.min(mahalanobis_distances_to_classes, dim=1)

        # Return negative distance so that higher score means more in-distribution
        return -min_distances

    def predict_ood_mahalanobis(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts OOD status based on Mahalanobis scores.

        Args:
            features (torch.Tensor): Input features of shape (B, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mahalanobis_scores (torch.Tensor): The computed scores (negative distances).
                - is_ood (torch.Tensor): Boolean tensor indicating OOD status (True if OOD).
        """
        mahalanobis_scores = self.get_mahalanobis_scores(features)
        # Higher score (less negative/more positive) means more in-distribution.
        # So, if score < threshold, it's OOD.
        is_ood = mahalanobis_scores < self.threshold
        return mahalanobis_scores, is_ood
