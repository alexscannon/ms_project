from typing import Tuple
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, f1_score

from src.loggers.wandb_logger import WandBLogger
from src.models.ood.ood_detector import OODDetector
from src.models.utils import extract_features_and_logits



class ContinualLearning:
    """
    Base class for continual learning methods.
    """

    def __init__(self, config: DictConfig, model: torch.nn.Module, device: torch.device):
        """
        Initialize the CL class with configuration, model, and device.

        Args:
            config (DictConfig): Configuration for the continual learning method.
            model (torch.nn.Module): The model to be used in continual learning.
            device (torch.device): The device on which the model will run.
        """
        self.config = config
        self.model = model
        self.device = device
        self.y_true_ood = []  # True labels for OOD data
        self.y_pred_ood = []  # Predicted labels for OOD data
        self.ind_total = 0
        self.ind_correct = 0


    def run_covariate_continual_learning(
            self,
            left_out_ind_dataloader: DataLoader,
            ood_dataloader: DataLoader,
            wandb_logger: WandBLogger,
            ood_detector: OODDetector,
            config: DictConfig,
            model: torch.nn.Module,
            checkpoint_class_info: dict,
            corrupted_dataloader: DataLoader | None = None,
        ) -> bool:
        """
        Runs a continual learning experiment on unlabeled streaming possibly covariate-shifted IND data and OOD data.

        Args:
            left_out_ind_dataloader (torch.utils.data.DataLoader): Dataloader for all the data that was not used while training.
            ood_dataloader (torch.utils.data.DataLoader): Dataloader for all OOD data.
            wandb_logger (WandBLogger): Logger for tracking metrics.
            ood_detector (OODDetector): OOD detector to determine whether the current example is OOD or IND.
            config (DictConfig): Configuration for the continual learning method.
            model (torch.nn.Module): The model to be used in continual learning.
            checkpoint_class_info (dict): Class information from the pre-training checkpoint data.
            corrupted_dataloader (torch.utils.data.DataLoader | None): Dataloader for corrupted data.
        Returns:
            None
        """
        # 1) Create a unified stream of IND and OOD data
        unified_stream = self._merge_streams(left_out_ind_dataloader, ood_dataloader, corrupted_dataloader)

        ind_classes = set(checkpoint_class_info.get("pretrain_classes", []))
        if config.data.name == "tiny_imagenet":
            ind_classes = set(left_out_ind_dataloader.dataset.targets) # type: ignore

        pbar = tqdm(unified_stream, desc="Continual learning: Processing data stream examples ...", unit="example")
        for batch_idx, (inputs, targets) in enumerate(pbar):

            # 1) Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 2) Extract features and logits
            features, logits = extract_features_and_logits(inputs, model)  # 2) Extract features and logits

            # 3) Predict OOD or IND
            is_ood = ood_detector.determine_ood_or_ind(logits, config.continual_learning.ood_method, features)
            is_ood_list = is_ood.tolist()  # Convert to list for easier handling
            self.y_pred_ood.extend(is_ood_list)  # Store the prediction for OOD detection

            # 4) Retrieve the True OOD labels
            targets_list = targets.tolist()  # Convert targets to list for easier handling
            if config.data.name == "cifar100":
                reverse_class_mapping = {new_cls: old_cls for old_cls, new_cls in checkpoint_class_info.get("pretrain_class_mapping", {}).items()}
                targets_is_ood_list = [1 if reverse_class_mapping.get(target) not in ind_classes else 0 for target in targets_list]
            else:
                targets_is_ood_list = [1 if target not in ind_classes else 0 for target in targets_list]
            self.y_true_ood.extend(targets_is_ood_list)  # Store the true label for OOD detection

            self.ind_total += (len(is_ood_list) - sum(is_ood_list))  # Count only IND examples that the model has determined as IND

            # 5) Make predictions, compute classification performance only on IND examples
            pred_y = torch.argmax(logits, dim=1)  # Get the predicted-labels from the logits [0 - 79]
            ind_mask = ~(is_ood.bool()) # Create a reverse mask for IND examples
            ind_pred_y = pred_y[ind_mask] # Filter targets to only include IND examples
            ind_targets = targets[ind_mask] # Filter targets to only include IND examples

            correct = (ind_pred_y == ind_targets).sum().item()  # Count correct predictions
            self.ind_correct += correct  # Update the count of correct predictions

            self.insert_into_buffer(logits, pred_y) # TODO: Update model/buffer accordingly

            if self.ind_total > self.config.continual_learning.warmup_metric_period:
                # TODO: Maybe break down by class classification performance
                wandb_logger.log({
                    "classification/accuracy": (self.ind_correct / self.ind_total) * 100,
                })

            if (batch_idx + 1) > self.config.continual_learning.warmup_metric_period: # Build a baseline for the first 100 examples
                f1, precision, ood_accuracy = self._compute_ood_metrics()

                wandb_logger.log({
                    "ood/current_perecision": precision,
                    "ood/current_f1": f1,
                    "ood/current_ood_accuracy": ood_accuracy
                })
        return True

    def _merge_streams(self, left_out_ind_dataloader: DataLoader, ood_dataloader: DataLoader, corrupted_dataloader: DataLoader | None) -> DataLoader:
        """
        Merges two dataloaders so that they are randomized and in a single dataloader.

        Args:
            left_out_ind_dataloader (torch.utils.data.DataLoader): Left-out training in-distribution dataloader.
            ood_dataloader (torch.utils.data.DataLoader): OOD dataloader.
            corrupted_dataloader (torch.utils.data.DataLoader | None): Corruption dataloader.
        Returns:
            unified_dataloader (torch.utils.data.DataLoader): The combined IND + OOD + (optional) corruption dataloader.
        """
        # Create a combined dataset from both dataloaders
        # combined_dataset = torch.utils.data.ConcatDataset([ind_dataloader.dataset])
        combined_dataset = torch.utils.data.ConcatDataset([left_out_ind_dataloader.dataset, ood_dataloader.dataset])
        if corrupted_dataloader:
            combined_dataset = torch.utils.data.ConcatDataset([combined_dataset, corrupted_dataloader.dataset])

        # TODO: Revert the dataset arg back combined_dataset below
        # Create a new DataLoader with the combined dataset
        unified_dataloader = DataLoader(
            dataset=combined_dataset, # type: ignore
            batch_size=self.config.continual_learning.batch_size,  # Currently set to 1 for streaming setting.
            shuffle=True,
            num_workers=left_out_ind_dataloader.num_workers
        )

        return unified_dataloader


    def _compute_ood_metrics(self) -> Tuple[float, float, float]:
        """
        Computes OOD detection metrics such as F1 score, precision, and accuracy.
        Args:
            None
        Returns:
            Tuple[float, float, float]: F1 score, precision, and accuracy.
        """
        # Leverage your OODDetector to compute AUROC
        inverted_y_true = 1 - np.array(self.y_true_ood)
        inverted_y_pred = 1 - np.array(self.y_pred_ood)  # Invert predictions for OOD detection

        accuracy = accuracy_score(self.y_true_ood, self.y_pred_ood)
        precision = precision_score(inverted_y_true, inverted_y_pred)
        f1 = f1_score(inverted_y_true, inverted_y_pred)

        return float(f1), float(precision), float(accuracy)


    def insert_into_buffer(self, inputs, pseudo_label):
        # Store or train on the new sample
        pass


    def initial_load_buffer(self, ind_left_in_dataloader):
        """
        Constructs initial buffer, loading each class with examples.
        """



