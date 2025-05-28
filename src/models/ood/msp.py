import torch
from omegaconf import DictConfig
import torch.nn.functional as F

class MSP:
    """
    Maximum Softmax Probability (MSP) method
    Paper: https://arxiv.org/pdf/1610.02136
    Why: Well-calibrated models tend to produce lower maximum probabilities for OOD samples.
    """

    def __init__(self, model: torch.nn.Module, ood_config: DictConfig):
        self.model = model
        self.ood_config = ood_config

    def predict_ood_msp(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a boolean decision if the examples are OOD or not.
        Args:
            logits (torch.Tensor): Logits of shape (B, C)
        Returns:
            is_ood (torch.Tensor): Boolean tensor of shape (B,)
            msp_scores (torch.Tensor): MSP scores of shape (B,)
        """
        msp_scores = self.get_msp_scores(logits)
        is_ood = msp_scores > self.ood_config.msp.threshold
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
