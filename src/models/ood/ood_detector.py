import logging
from typing import Tuple
import torch
from omegaconf import DictConfig
from src.models.ood.msp import MSP
from src.models.ood.odin import ODINDetector
from src.models.ood.mahalanobis import MahalanobisDetector
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from src.models.ood.energy import EnergyDetector
from src.models.ood.knn import KNNDetector
from src.utils import plot_roc_curves

class OODDetector:
    """
    OOD detector class supporting multiple OOD detection methods.
    """

    def __init__(self, config: DictConfig, model: torch.nn.Module, device: str = 'cuda', temperature: float = 1.0):
        self.config = config
        self.model = model
        self.device = device
        self.temperature = temperature # Temperature for the softmax function

        self.model.to(self.device) # Move model to device
        self.model.eval() # Set model to evaluation mode

        # Initialize all detectors
        self.detectors = {
            "msp": MSP(self.model, self.config.ood),
            "odin": ODINDetector(self.model, self.config.ood),
            "mahalanobis": MahalanobisDetector(self.model, self.config.ood, device=self.device),
            "energy": EnergyDetector(self.model, self.config.ood),
            "knn": KNNDetector(self.model, self.config.ood),
        }


    def run_ood_detection(self, left_out_ind_dataloader: torch.utils.data.DataLoader, ood_dataloader: torch.utils.data.DataLoader, pretrained_ind_dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Run OOD detection on the remaining in-distribution and out-of-distribution dataloaders.
        Args:
            left_out_ind_dataset (torch.utils.data.Dataset): In-distribution dataset
            ood_dataset (torch.utils.data.Dataset): Out-of-distribution dataset
            pretrained_ind_dataset (torch.utils.data.Dataset): Pretrained in-distribution dataset
        Returns:
            dict: Dictionary containing the OOD detection scores and labels
        """
        # Fit the Mahalanobis detector
        self.detectors["mahalanobis"].fit(
            pretrained_ind_dataloader,
            self.extract_features_and_logits,
            self.config.data.num_ind_classes
        )

        # Fit the KNN detector
        self.detectors["knn"].fit(
            pretrained_ind_dataloader,
            self.extract_features_and_logits,
            self.config.data.num_ind_classes
        )

        left_out_ind_stats, ood_stats = self.compute_all_ood_stats(left_out_ind_dataloader, ood_dataloader)
        # Plot ROC curves
        aurocs: dict = self.calculate_all_aurocs(left_out_ind_stats, ood_stats)
        logging.info(f"Plotting ROC curves...")
        plot_roc_curves(left_out_ind_stats, ood_stats, detector_names=list(aurocs.keys()))
        return aurocs

    def compute_all_ood_stats(self, left_out_ind_dataloader: torch.utils.data.DataLoader, ood_dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Compute all OOD detection scores for the left-out in-distribution and out-of-distribution datasets.
        """
        logging.info("Computing OOD detection scores for left-out in-distribution dataset...")
        left_out_ind_stats: dict = self.compute_data_ood_stats(left_out_ind_dataloader)
        logging.info("Computing OOD detection scores for out-of-distribution dataset...")
        ood_stats: dict = self.compute_data_ood_stats(ood_dataloader)
        return left_out_ind_stats, ood_stats


    def calculate_all_aurocs(self, left_out_ind_stats: dict, ood_stats: dict) -> dict:
        """
        Calculate all AUROCs for the OOD detection methods.
        Args:
            left_out_ind_stats (dict): Dictionary containing the OOD detection scores and other relevant statistics for the left-out in-distribution dataset
            ood_stats (dict): Dictionary containing the OOD detection scores and other relevant statistics for the out-of-distribution dataset
        Returns:
            dict: Dictionary containing the AUROCs for the OOD detection methods
        """
        auroc_msp = None
        auroc_odin = None
        auroc_mahalanobis = None
        auroc_energy = None
        auroc_knn = None

        if "msp" in self.detectors:
            auroc_msp = self.evaluate_with_auroc(left_out_ind_stats, ood_stats, "msp")
        if "odin" in self.detectors:
            auroc_odin = self.evaluate_with_auroc(left_out_ind_stats, ood_stats, "odin")
        if "mahalanobis" in self.detectors:
            auroc_mahalanobis = self.evaluate_with_auroc(left_out_ind_stats, ood_stats, "mahalanobis")
        if "energy" in self.detectors:
            auroc_energy = self.evaluate_with_auroc(left_out_ind_stats, ood_stats, "energy")
        if "knn" in self.detectors:
            auroc_knn = self.evaluate_with_auroc(left_out_ind_stats, ood_stats, "knn")

        return {
            "msp": auroc_msp,
            "odin": auroc_odin,
            "mahalanobis": auroc_mahalanobis,
            "energy": auroc_energy,
            "knn": auroc_knn
        }


    def compute_data_ood_stats(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Compute all scores for the OOD detection methods.

        Args:
            dataloader (torch.utils.data.DataLoader): Input dataloader
        Returns:
            dict: Dictionary containing the OOD detection scores and labels
        """
        batch_cache = {
            "batch_data_info": {
                "logits": [],
                "features": [],
                "labels": [],
            },
            "detector_scores": {
                "odin_scores": [],
                "msp_scores": [],
                "mahalanobis_scores": [],
                "energy_scores": [],
                "knn_scores": [],
            }
        }

        pbar = tqdm(dataloader, desc=f"Computing OOD detection scores ...")
        for _, batch in enumerate(pbar): # Iterate over the dataloader, tuple is (batch_idx, batch)
            if isinstance(batch, (list, tuple)): # Batch has two elements: x (input) and y (labels)
                x, y = batch[0], batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None # If no labels are provided, set y to None

            # Move data to device for faster computation
            x = x.to(self.device)
            if y is not None:
                y = y.to(self.device)
                batch_cache["batch_data_info"]["labels"].append(y)

            # Compute logits and features
            with torch.no_grad():
                logits, features = self.extract_features_and_logits(x)
                batch_cache["batch_data_info"]["logits"].append(logits)
                batch_cache["batch_data_info"]["features"].append(features)

            # =========== Compute OOD scores =========== #
            # ODIN
            if "odin" in self.detectors:
                odin_scores = self.detectors["odin"].get_odin_scores(x, y)
                batch_cache["detector_scores"]["odin_scores"].append(odin_scores)

            # MSP
            if "msp" in self.detectors:
                msp_scores = self.detectors["msp"].get_msp_scores(logits)
                batch_cache["detector_scores"]["msp_scores"].append(msp_scores)

            # Energy
            if "energy" in self.detectors:
                energy_scores = self.detectors["energy"].get_energy_scores(logits)
                batch_cache["detector_scores"]["energy_scores"].append(energy_scores)

            # Mahalanobis
            if "mahalanobis" in self.detectors:
                mahalanobis_detector = self.detectors["mahalanobis"]
                if mahalanobis_detector.is_fitted:
                    mahalanobis_scores = mahalanobis_detector.get_mahalanobis_scores(features)
                    batch_cache["detector_scores"]["mahalanobis_scores"].append(mahalanobis_scores)
                elif not hasattr(mahalanobis_detector, "_warned_not_fitted"): # Log warning only once
                    logging.warning("Mahalanobis detector is not fitted. Skipping Mahalanobis score computation for this dataloader.")
                    mahalanobis_detector._warned_not_fitted = True # Prevent repeated warnings

            # KNN
            if "knn" in self.detectors:
                knn_detector = self.detectors["knn"]
                if knn_detector.is_fitted:
                    knn_scores = knn_detector.get_knn_scores(features)
                    batch_cache["detector_scores"]["knn_scores"].append(knn_scores)
                elif not hasattr(knn_detector, "_warned_not_fitted"): # Log warning only once
                    logging.warning("KNN detector is not fitted. Skipping KNN score computation for this dataloader.")
                    knn_detector._warned_not_fitted = True # Prevent repeated warnings

        all_logits = torch.cat(batch_cache["batch_data_info"]["logits"], dim=0) if batch_cache["batch_data_info"]["logits"] else torch.empty(0)
        all_features = torch.cat(batch_cache["batch_data_info"]["features"], dim=0) if batch_cache["batch_data_info"]["features"] else torch.empty(0)
        all_labels = torch.cat(batch_cache["batch_data_info"]["labels"], dim=0) if batch_cache["batch_data_info"]["labels"] else torch.empty(0)

        all_odin_scores = None
        if batch_cache["detector_scores"]["odin_scores"]:
            all_odin_scores = torch.cat(batch_cache["detector_scores"]["odin_scores"], dim=0)

        all_msp_scores = None
        if batch_cache["detector_scores"]["msp_scores"]:
            all_msp_scores = torch.cat(batch_cache["detector_scores"]["msp_scores"], dim=0)

        all_mahalanobis_scores = None
        if batch_cache["detector_scores"]["mahalanobis_scores"]:
            all_mahalanobis_scores = torch.cat(batch_cache["detector_scores"]["mahalanobis_scores"], dim=0)

        all_energy_scores = None
        if batch_cache["detector_scores"]["energy_scores"]:
            all_energy_scores = torch.cat(batch_cache["detector_scores"]["energy_scores"], dim=0)

        all_knn_scores = None
        if batch_cache["detector_scores"]["knn_scores"]:
            all_knn_scores = torch.cat(batch_cache["detector_scores"]["knn_scores"], dim=0)

        results = {
            "all_logits": all_logits,
            "all_features": all_features,
            "all_labels": all_labels,
        }
        if all_odin_scores is not None:
            results["all_odin_scores"] = all_odin_scores
        if all_msp_scores is not None:
            results["all_msp_scores"] = all_msp_scores
        if all_mahalanobis_scores is not None:
            results["all_mahalanobis_scores"] = all_mahalanobis_scores
        if all_energy_scores is not None:
            results["all_energy_scores"] = all_energy_scores
        if all_knn_scores is not None:
            results["all_knn_scores"] = all_knn_scores

        return results


    def extract_features_and_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and logits from the model.

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            features (torch.Tensor): Features from the model
            logits (torch.Tensor): Logits from the model
        """
        self.model.eval() # Set model to evaluation mode
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

    def evaluate_with_auroc(self, left_out_ind_stats: dict, ood_stats: dict, detector_name: str) -> float:
        """
        Evaluate using AUROC (threshold-independent)
        Args:
            left_out_ind_stats (dict): Dictionary containing the OOD detection scores and other relevant statistics for the left-out in-distribution dataset
            ood_stats (dict): Dictionary containing the OOD detection scores and other relevant statistics for the out-of-distribution dataset
            detector_name (str): Name of the detector to evaluate
        Returns:
            auroc (float): Area under the ROC curve
        """
        logging.info(f"Calculating AUROCs for the {detector_name} detector...")
        key_name = f"all_{detector_name}_scores"

        # Ensure scores are present and are tensors
        ind_scores_tensor = left_out_ind_stats.get(key_name)
        ood_scores_tensor = ood_stats.get(key_name)

        if ind_scores_tensor is None or ood_scores_tensor is None:
            logging.warning(f"Scores for '{detector_name}' not found in one or both datasets for AUROC calculation. Skipping.")
            return np.nan # Return NaN or handle appropriately

        if not isinstance(ind_scores_tensor, torch.Tensor) or not isinstance(ood_scores_tensor, torch.Tensor):
            logging.warning(f"Scores for '{detector_name}' are not tensors. Skipping AUROC calculation.")
            return np.nan

        # Convert to numpy for sklearn
        ind_scores = ind_scores_tensor.cpu().numpy()
        ood_scores = ood_scores_tensor.cpu().numpy()

        all_scores = np.concatenate([ind_scores, ood_scores])
        logging.info(f"Shape of all scores: {all_scores.shape}")

        # Create labels: 0 for in-distribution, 1 for out-of-distribution
        # Note: The original code used labels from the data, which might be class labels.
        # For OOD AUROC, we need binary labels indicating ID vs OOD.
        # Let's assume left_out_ind_stats are ID (label 0) and ood_stats are OOD (label 1).
        labels_ind = np.zeros(len(ind_scores), dtype=int)
        labels_ood = np.ones(len(ood_scores), dtype=int)
        all_labels = np.concatenate([labels_ind, labels_ood])
        logging.info(f"Shape of all labels: {all_labels.shape}")

        # For AUROC calculation, roc_auc_score expects higher scores for the positive class (OOD).
        # Our OOD scores are designed such that:
        # - MSP: higher = ID
        # - ODIN: higher = ID (max softmax probability after perturbation)
        # - Mahalanobis: higher = ID (negative distance)
        # So, for all these, lower scores indicate OOD.
        # roc_auc_score(y_true, y_score) where y_score is confidence score for class 1 (OOD).
        # If our scores are confidence for ID, then we use 1 - score or -score for OOD confidence.
        # Using -all_scores effectively flips it so lower original scores (more OOD) become higher values.
        try:
            auroc = roc_auc_score(all_labels, -all_scores)
        except ValueError as e:
            logging.error(f"AUROC calculation failed for {detector_name}: {e}. Scores: {all_scores[:10]}, Labels: {all_labels[:10]}")
            auroc = np.nan # Return NaN if calculation fails (e.g. only one class present in y_true)
        return auroc

    # def get_ood_scores(self, left_out_ind_dataloader: torch.utils.data.DataLoader, ood_dataloader: torch.utils.data.DataLoader) -> dict:
    #     """
    #     Get OOD scores from the input tensor.
    #     Args:
    #         left_out_ind_dataloader (torch.utils.data.DataLoader): In-distribution dataset
    #         ood_dataloader (torch.utils.data.DataLoader): Out-of-distribution dataset
    #     Returns:
    #         dict: Dictionary containing the OOD detection scores and labels
    #     """
    #     # x_ind, y_ind = self._ensure_batch_format(left_out_ind_dataloader)
    #     # x_ood, y_ood = self._ensure_batch_format(ood_dataloader)

    #     # Extract features and logits
    #     features_ind, logits_ind = self.extract_features_and_logits(left_out_ind_dataloader)
    #     features_ood, logits_ood = self.extract_features_and_logits(ood_dataloader)

    #     # ================ MSP ================ #
    #     msp = MSP(self.model, self.config.ood)
    #     msp_scores_ind, is_ood_msp_ind = msp.predict_ood_msp(logits_ind)
    #     msp_scores_ood, is_ood_msp_ood = msp.predict_ood_msp(logits_ood)

    #     # ================ ODIN ================ #
    #     odin = ODINDetector(self.model, self.config.ood)
    #     odin_scores_ind, is_ood_odin_ind = odin.predict_ood_odin(left_out_ind_dataloader)
    #     odin_scores_ood, is_ood_odin_ood = odin.predict_ood_odin(ood_dataloader)

    #     return {
    #         "msp": {
    #             "scores_ind": msp_scores_ind,
    #             "is_ood_ind": is_ood_msp_ind,
    #             "scores_ood": msp_scores_ood,
    #             "is_ood_ood": is_ood_msp_ood
    #         },
    #         "odin": {
    #             "scores_ind": odin_scores_ind,
    #             "is_ood_ind": is_ood_odin_ind,
    #             "scores_ood": odin_scores_ood,
    #             "is_ood_ood": is_ood_odin_ood
    #         }
    #     }


    # def _ensure_batch_format(self, input: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Ensure inputs have batch dimension.

    #     Args:
    #         input (torch.utils.data.DataLoader): Input dataloader

    #     Returns:
    #         x (torch.Tensor): Shape [B, C, H, W]
    #         y (torch.Tensor): Shape [B] or None
    #     """
    #     # Add batch dimension if needed
    #     logging.info(f"Input type: {type(input)}")
    #     logging.info(f"Input dataset: {input.dataset}")
    #     x, y = input.dataset
    #     if x.ndim == 3:
    #         x = x.unsqueeze(0)

    #     # Handle labels
    #     if y is not None:
    #         if isinstance(y, int):
    #             y = torch.tensor([y], device=x.device)
    #         elif y.ndim == 0:  # scalar tensor
    #             y = y.unsqueeze(0)

    #     return x, y # Return tuple of (x, y)



