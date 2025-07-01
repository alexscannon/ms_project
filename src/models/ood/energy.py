import torch
import torch.nn.functional as F


class EnergyDetector:
    def __init__(self, model, ood_config):
        """
        Energy detector class.
        Paper: https://arxiv.org/pdf/2010.03759
        Args:
            model (torch.nn.Module): The model to use for energy detection.
            ood_config (DictConfig): The configuration for the energy detector.
        """
        self.model = model
        self.ood_config = ood_config
        self.temperature = getattr(ood_config.energy, 'temperature', 1.0)

    def predict_ood_energy(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the energy score for the input logits and a boolean decision if the input is OOD or not.

        Args:
            logits (torch.Tensor): Model logits with shape [batch_size, num_classes]

        Returns:
            tuple: (energy_scores, is_ood)
                energy_scores (torch.Tensor): Energy scores of shape [batch_size]
                is_ood (torch.Tensor): Boolean tensor indicating if each example is OOD
        """
        energy_scores = self.get_energy_scores(logits)
        is_ood = energy_scores < getattr(self.ood_config.energy, "threshold", 0.0)  # OOD if energy score is negative
        return energy_scores, is_ood

    def get_energy_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy scores for the given logits.

        Energy function: E(x) = -T * log(sum(exp(f_i(x)/T)))
        where f_i(x) are the logits and T is temperature.

        For OOD detection, we use negative energy scores since:
        - Higher energy (more negative energy score) = more in-distribution
        - Lower energy (less negative energy score) = more out-of-distribution

        Args:
            logits (torch.Tensor): Model logits with shape [batch_size, num_classes]

        Returns:
            torch.Tensor: Energy scores with shape [batch_size]
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute energy: E(x) = -T * log(sum(exp(f_i(x)/T)))
        # Using logsumexp for numerical stability
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=1)

        # Return negative energy for OOD detection
        # Higher values indicate more confidence (in-distribution)
        # Lower values indicate less confidence (out-of-distribution)
        return -energy
