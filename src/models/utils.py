import torch
from typing import Tuple
import logging

def extract_features_and_logits(x: torch.Tensor, model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and logits from the model.

    Args:
        x (torch.Tensor): Input tensor
    Returns:
        tuple: A tuple containing:
            features (torch.Tensor): 'Features from the model'
            logits (torch.Tensor): 'Logits from the model'
    """
    model.eval() # Set model to evaluation mode
    logits = model(x)
    # For ViT, we can extract features from the last layer before classification
    # This assumes the model has a 'head' attribute for the classification layer
    if hasattr(model, 'head'): # TODO: Change this to "mlp_head" for custom ViT model
        features = model.forward_features(x)
        if hasattr(features, 'shape') and len(features.shape) > 2:
            features = features.mean(dim=1)  # Global average pooling if needed
    else:
        # Fallback: use logits as features
        features = logits

    return features, logits