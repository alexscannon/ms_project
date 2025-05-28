from typing import Tuple
import torch
from omegaconf import DictConfig
from src.models.ood.msp import MSP
from src.models.ood.odin import ODINDetector
from sklearn.metrics import roc_auc_score
import numpy as np

class OODDetector:
    """
    OOD detector class supporting multiple OOD detection methods.
    """

    def __init__(
            self,
            config: DictConfig,
            model: torch.nn.Module,
            device: str = 'cuda',
            temperature: float = 1.0,
        ):
        self.config = config
        self.model = model
        self.temperature = temperature # Temperature for the softmax function
        self.device = device

        self.model.to(self.device) # Move model to device
        self.model.eval() # Set model to evaluation mode


    def extract_features_and_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and logits from the model.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W)
        Returns:
            features (torch.Tensor): Features from the model
            logits (torch.Tensor): Logits from the model
        """
        with torch.no_grad():
            logits = self.model(x)
            # For ViT, we can extract features from the last layer before classification
            # This assumes the model has a 'head' attribute for the classification layer
            if hasattr(self.model, 'head'):
                features = self.model.forward_features(x)
                if hasattr(features, 'shape') and len(features.shape) > 2:
                    features = features.mean(dim=1)  # Global average pooling if needed
            else:
                # Fallback: use logits as features
                features = logits

        return features, logits


    def run_ood_detection(self, left_out_ind_dataset: torch.utils.data.Dataset, ood_dataset: torch.utils.data.Dataset) -> dict:
        """
        Run OOD detection on the input tensor.
        Args:
            left_out_ind_dataset (torch.utils.data.Dataset): In-distribution dataset
            ood_dataset (torch.utils.data.Dataset): Out-of-distribution dataset
        Returns:
            dict: Dictionary containing the OOD detection scores and labels
        """
        scores: dict = self.get_ood_scores(left_out_ind_dataset, ood_dataset)
        aurocs: dict = self.calculate_all_aurocs(scores)
        return aurocs

    def get_mahalanobis_distance(self, input: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance method"""
        pass

    def get_energy_score(self, input: torch.Tensor) -> torch.Tensor:
        """Energy score method"""
        pass

    def get_entropy_score(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def calculate_all_aurocs(self, scores: dict) -> dict:
        """
        Calculate all AUROCs for the OOD detection methods.
        Args:
            ood_scores (dict): Dictionary containing the OOD detection scores and labels
        Returns:
            dict: Dictionary containing the AUROCs for the OOD detection methods
        """
        auroc_msp = self.evaluate_with_auroc(scores["msp"]["scores_ind"], scores["msp"]["scores_ood"])
        auroc_odin = self.evaluate_with_auroc(scores["odin"]["scores_ind"], scores["odin"]["scores_ood"])

        return {
            "msp": auroc_msp,
            "odin": auroc_odin
        }


    def evaluate_with_auroc(self, ind_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
        """
        Evaluate using AUROC (threshold-independent)
        Args:
            ind_scores (torch.Tensor): Scores for in-distribution samples
            ood_scores (torch.Tensor): Scores for out-of-distribution samples
        Returns:
            auroc (float): Area under the ROC curve
        """
        all_scores = np.array(ind_scores.cpu().numpy() + ood_scores.cpu().numpy())
        all_labels = np.array([0] * len(ind_scores.cpu().numpy()) + [1] * len(ood_scores.cpu().numpy()))

        # For AUROC calculation, we need to flip scores since lower = OOD
        auroc = roc_auc_score(all_labels, -all_scores)  # Negative because lower scores = OOD
        return auroc

    def get_ood_scores(self, left_out_ind_dataset: torch.utils.data.Dataset, ood_dataset: torch.utils.data.Dataset) -> dict:
        """
        Get OOD scores from the input tensor.
        Args:
            left_out_ind_dataset (torch.utils.data.Dataset): In-distribution dataset
            ood_dataset (torch.utils.data.Dataset): Out-of-distribution dataset
        Returns:
            dict: Dictionary containing the OOD detection scores and labels
        """
        x_ind, y_ind = self._ensure_batch_format(left_out_ind_dataset)
        x_ood, y_ood = self._ensure_batch_format(ood_dataset)

        # Extract features and logits
        features_ind, logits_ind = self.extract_features_and_logits(x_ind)
        features_ood, logits_ood = self.extract_features_and_logits(x_ood)

        # ================ MSP ================ #
        msp = MSP(self.model, self.config.ood)
        msp_scores_ind, is_ood_msp_ind = msp.predict_ood_msp(logits_ind)
        msp_scores_ood, is_ood_msp_ood = msp.predict_ood_msp(logits_ood)

        # ================ ODIN ================ #
        odin = ODINDetector(self.model, self.config.ood)
        odin_scores_ind, is_ood_odin_ind = odin.predict_ood_odin(x_ind, y_ind)
        odin_scores_ood, is_ood_odin_ood = odin.predict_ood_odin(x_ood, y_ood)

        return {
            "msp": {
                "scores_ind": msp_scores_ind,
                "is_ood_ind": is_ood_msp_ind,
                "scores_ood": msp_scores_ood,
                "is_ood_ood": is_ood_msp_ood
            },
            "odin": {
                "scores_ind": odin_scores_ind,
                "is_ood_ind": is_ood_odin_ind,
                "scores_ood": odin_scores_ood,
                "is_ood_ood": is_ood_odin_ood
            }
        }



    def _ensure_batch_format(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure inputs have batch dimension.

        Args:
            input (torch.Tensor): Input tensor, shape [C, H, W] or [B, C, H, W]

        Returns:
            x (torch.Tensor): Shape [B, C, H, W]
            y (torch.Tensor): Shape [B] or None
        """
        # Add batch dimension if needed
        x, y = input
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Handle labels
        if y is not None:
            if isinstance(y, int):
                y = torch.tensor([y], device=x.device)
            elif y.ndim == 0:  # scalar tensor
                y = y.unsqueeze(0)

        return x, y # Return tuple of (x, y)


