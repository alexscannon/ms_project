import torch
from omegaconf import DictConfig
import torch.nn.functional as F
import logging
from tqdm import tqdm

class MSP:
    """
    Maximum Softmax Probability (MSP) method
    Paper: https://arxiv.org/pdf/1610.02136
    Why: Well-calibrated models tend to produce lower maximum probabilities for OOD samples.
    """

    def __init__(self, model: torch.nn.Module, ood_config: DictConfig):
        self.model = model
        self.ood_config = ood_config
        self.is_fitted = False

    def fit(self, dataloader: torch.utils.data.DataLoader, extract_fn, *args):
        """
        Calibrate the threshold on in-distribution data.
        Sets the threshold to the q-th percentile of MSP scores on the provided data.
        """
        logging.info("Calibrating MSP threshold...")
        all_msp_scores = []
        pbar = tqdm(dataloader, desc="Extracting MSP scores...", unit="batch")
        for batch in pbar:
            x, _ = batch
            x = x.to(next(self.model.parameters()).device)
            with torch.no_grad():
                _, logits = extract_fn(x, self.model)
            msp_scores = self.get_msp_scores(logits)
            all_msp_scores.append(msp_scores.cpu())

        all_msp_scores = torch.cat(all_msp_scores)

        # A common approach is to set the threshold to correctly classify 95% of ID data.
        # Since lower scores indicate OOD, the threshold is the 5th percentile.
        q = 1.0 - self.ood_config.msp.get("id_confidence_percentile", 0.95)
        self.threshold = torch.quantile(all_msp_scores, q).item()
        self.is_fitted = True
        logging.info(f"MSP threshold calibrated to: {self.threshold:.4f}")


    def predict_ood_msp(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a boolean decision if the examples are OOD or not.

        Args:
            logits (torch.Tensor): Logits of shape (B, C).
        Returns:
            is_ood (torch.Tensor): Boolean tensor of shape (B,).
            msp_scores (torch.Tensor): MSP scores of shape (B,).
        """
        msp_scores = self.get_msp_scores(logits)
        is_ood = (msp_scores < self.threshold).int()  # Convert to int for boolean representation
        return msp_scores, is_ood

    def get_msp_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the Maximum Softmax Probability (MSP) for examples.
        Args:
            logits (torch.Tensor): Logits of shape (B, C)
        Returns:
            msp_scores (torch.Tensor): MSP scores of shape (B,)
        """
        softmax_scores = F.softmax(logits, dim=1) # Convert logits to probabilties
        msp_scores, _ = torch.max(softmax_scores, dim=1) # Get the maximum probability for the sample
        return msp_scores
