import torch
from omegaconf import DictConfig
import torch.nn.functional as F

class ODINDetector:
    """
    ODIN detector class
    Paper: https://arxiv.org/pdf/1706.02690
    Why: ODIN is a method that uses the temperature scaling technique to calibrate the model's confidence scores.

    ODIN uses:
        1. Temperature scaling to soften the softmax distribution
        2. Input preprocessing (adding small perturbations) to increase the separation
        between in-distribution and out-of-distribution samples

    ODIN Process:
        1. Temperature scale the logits to magnify lukewarm confidence scores
        2. Input preprocessing: Apply an adversarial perturbation to the input tensor in the direction of the gradient
            of the loss function with epsilon as the magnitude of the perturbation.
        3. Compute the score as the difference between the temperature-scaled logits and the original logits
        4. Return the score as the OOD score
    """

    def __init__(self, model: torch.nn.Module, ood_config: DictConfig, threshold: float = 0.5):
        self.model = model
        self.ood_config = ood_config

        # Hyperparameters
        self.temperature = ood_config.odin.temperature
        self.epsilon = ood_config.odin.epsilon
        self.threshold = ood_config.odin.threshold
        self.criterion = ood_config.odin.criterion

        self.model.eval() # Set model to evaluation mode

    def predict_ood_odin(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the OODIN score for the input tensor and a boolean decision if the input is OOD or not.

        Args:
            logits (torch.Tensor): predicted logits from the model
        Returns:
            odin_score (torch.Tensor): OODIN score of shape (B,)
            is_ood (torch.Tensor): Boolean tensor of shape (B,)
        """
        odin_score = self.get_odin_scores(logits)
        is_ood = odin_score < self.threshold
        return odin_score, is_ood

    def get_odin_scores(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Retrieve OODIN score for the input tensor.

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor | None): Ground truth labels
        Returns:
            scores (torch.Tensor): ODIN score of shape (B,)
        """
        preprocessed_input = self.preprocess_inputs(x, y)
        with torch.no_grad():
            logits = self.model(preprocessed_input)
            probs = logits.softmax(dim=1)
            max_probs, _ = probs.max(dim=1)
        return max_probs

    def preprocess_inputs(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply adversarial perturbation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor | None): Ground truth labels
        Returns:
            preprocessed_input (torch.Tensor): Preprocessed input tensor
        """
        with torch.enable_grad(): # explicity turn on gradient computation
            x = x.clone().detach() # Clone the input tensor so that we don't modify the original tensor
            x.requires_grad = True # Compute gradients for the cloned input tensor in order to compute the gradient vector

            # Forward pass
            logits = self.model(x) / self.temperature # Temperature scaling

            # Compute the predicted labels
            predicted_labels = torch.argmax(logits, dim=1)

            # Compute the loss
            if self.criterion == "NLL":
                # loss = F.nll_loss(logits, y) TODO: Look into if this is correct
                loss = F.nll_loss(F.log_softmax(logits, dim=1), predicted_labels) # ODIN uses the predicted labels to compute the loss
            elif self.criterion == "CE":
                loss = F.cross_entropy(logits, predicted_labels)
            else:
                raise ValueError(f"Invalid criterion: {self.criterion}")


            # Backward pass to compute gradients
            loss.backward()
            gradient_vector = x.grad.sign() if x.grad is not None else torch.zeros_like(x) # Sign of the gradient

            # Apply the perturbation (subtract to increase confidence)
            preprocessed_input = x - (self.epsilon * gradient_vector)

        return preprocessed_input.detach() # Return tensor with gradients

    def calibrate_odin(self) -> torch.Tensor:
        """
        Calibrate the ODIN model's threshold.
        """
        raise NotImplementedError("Calibrating the ODIN model's threshold is not implemented as it is not needed.")
