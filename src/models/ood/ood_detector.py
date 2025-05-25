import torch
from omegaconf import DictConfig
import torch.nn.functional as F

class OODDetector:
    """
    OOD detector class supporting multiple OOD detection methods.
    """

    def __init__(
            self,
            config: DictConfig,
            model: torch.nn.Module,
            temperature: float = 1.0,
            device: str = 'cuda',
            left_out_ind_dataset: torch.utils.data.Dataset = None,
            ood_dataset: torch.utils.data.Dataset = None
        ):
        self.config = config
        self.model = model
        self.temperature = temperature # Temperature for the softmax function
        self.device = device

        self.left_out_ind_dataset = left_out_ind_dataset
        self.ood_dataset = ood_dataset

        self.model.to(self.device) # Move model to device
        self.model.eval() # Set model to evaluation mode


    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from the model.
        """

    def get_msp_score(self, inputs: torch.Tensor) -> torch.Tensor:
        """Maximum Softmax Probability (MSP) method"""
        pass

    def get_max_logit(self, inputs: torch.Tensor) -> torch.Tensor:
        """Maximum logit method"""
        pass

    def get_mahalanobis_distance(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance method"""
        pass

    def get_energy_score(self, inputs: torch.Tensor) -> torch.Tensor:
        """Energy score method"""
        pass

    def get_entropy_score(self, inputs: torch.Tensor) -> torch.Tensor:
        """Entropy score method"""
        pass

