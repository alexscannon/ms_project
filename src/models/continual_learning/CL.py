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
            checkpoint_class_info: dict
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
        Returns:
            None
        """
        # 1) Create a unified stream of IND and OOD data
        unified_stream = self._merge_streams(left_out_ind_dataloader, ood_dataloader)
        ind_classes = set(checkpoint_class_info.get("pretrain_classes", []))

        example_counter = 0
        pbar = tqdm(unified_stream, desc="Continual learning: Processing data stream examples ...", unit="example")
        for example_idx, (input, targets) in enumerate(pbar):

            input = input.to(self.device) # 1) Move data to device
            features, logits = extract_features_and_logits(input, model)  # 2) Extract features and logits

            example_counter += 1

            # 3) Determine OOD or IND
            is_ood = ood_detector.determine_ood_or_ind(logits, config.ood_method, features)
            self.y_pred_ood.append(int(is_ood))  # Store the prediction for OOD detection
            self.y_true_ood.append(1 if np.int64(targets.item()) not in ind_classes else 0)  # Store the true label for OOD detection

            # 4) Assign pseudo-label if IND
            if not is_ood:
                self.ind_total += 1
                pseudo_label = torch.argmax(logits, dim=1).item()  # Get the pseudo-label from the logits
                self.insert_into_buffer(logits, pseudo_label) # TODO: Update model/buffer accordingly

                if self.ind_total > self.config.continual_learning.warmup_metric_period:
                    if pseudo_label == targets.item():
                        self.ind_correct += 1
                    # TODO: Maybe break down by class classification performance
                    wandb_logger.log({
                        "classification/accuracy": (self.ind_correct / self.ind_total) * 100,
                    })

            # 4) Every 100 examples: compute detection AUROC & classification performance, then log to W&B
            if example_counter > self.config.continual_learning.warmup_metric_period: # Build a baseline for the first 100 examples
                f1, precision, ood_accuracy = self._compute_ood_metrics()

                wandb_logger.log({
                    "ood/current_perecision": precision,
                    "ood/current_f1": f1,
                    "ood/current_ood_accuracy": ood_accuracy,
                    "stream_step": example_counter,
                })
        return True

    def _merge_streams(self, ind_dataloader: DataLoader, ood_dataloader: DataLoader) -> DataLoader:
        """
        Merges two dataloaders so that they are randomized and in a single dataloader.

        Args:
            ind_dataloader (torch.utils.data.DataLoader): In-distribution dataloader.
            ood_dataloader (torch.utils.data.DataLoader): OOD dataloader.
        Returns:
            unified_dataloader (torch.utils.data.DataLoader): The combined IND + OOD dataloader.
        """
        # Create a combined dataset from both dataloaders
        # combined_dataset = torch.utils.data.ConcatDataset([ind_dataloader.dataset, ood_dataloader.dataset])
        combined_dataset = torch.utils.data.ConcatDataset([ind_dataloader.dataset])


        # Create a new DataLoader with the combined dataset
        unified_dataloader = DataLoader(
            combined_dataset,
            batch_size=self.config.continual_learning.batch_size,  # Currently set to 1 for streaming setting.
            shuffle=True,
            num_workers=ind_dataloader.num_workers
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
        precision = precision_score(self.y_true_ood, self.y_pred_ood, average='binary')
        accuracy = accuracy_score(self.y_true_ood, self.y_pred_ood)
        f1 = f1_score(self.y_true_ood, self.y_pred_ood, average='binary')

        return float(f1), float(precision), float(accuracy)


    def insert_into_buffer(self, inputs, pseudo_label):
        # Store or train on the new sample
        pass


    def initial_load_buffer(self, ind_left_in_dataloader):
        """
        Constructs initial buffer, loading each class with examples.
        """



