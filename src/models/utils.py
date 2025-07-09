import torch
from typing import Tuple
import logging
from omegaconf import DictConfig

def get_feature_dim(model: torch.nn.Module, config: DictConfig, device: torch.device) -> int:

    try:
        # For Timm models
        return model.num_features # type: ignore
    except AttributeError:
        # Fallback for other models
        dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size).to(device)
        features, _ = extract_features_and_logits(dummy_input, model)
        return features.shape[1]

def extract_features_and_logits(x: torch.Tensor, model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and logits from the model.

    Args:
        x (torch.Tensor): Input tensor
    Returns:
        features (torch.Tensor): Features from the model.
        logits (torch.Tensor): Logits from the model.
    """
    model.eval() # Set model to evaluation mode
    logits = model(x)
    # For ViT, we can extract features from the last layer before classification
    # This assumes the custom model has a 'mlp_head' or timm model has attribute for the classification layer
    if hasattr(model, 'mlp_head') or hasattr(model, 'head'):
        features = model.forward_features(x) # type: ignore
        if hasattr(features, 'shape') and len(features.shape) > 2:
            features = features.mean(dim=1)  # Global average pooling if needed
    else:
        # Fallback: use logits as features
        logging.warning("Model does not have attribute head/mlp_head to extract feature from! Falling back to logits.")
        features = logits

    return features, logits