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


def assign_to_clusters(embeddings: torch.Tensor, ood_threshold: float, num_classes: int, device: torch.device, stream_clusters) -> torch.Tensor:
    """
    Assigns each embedding to the nearest cluster or to OOD (-1) based on Mahalanobis distance.

    Args:
        embeddings: Tensor of shape [batch_size, feature_dim] containing the embeddings
        ood_threshold: Maximum Mahalanobis distance threshold for in-distribution assignment

    Returns:
        Tensor of shape [batch_size] with cluster assignments (-1 for OOD)
    """
    batch_size = embeddings.shape[0]

    # Initialize tensor to store distances to each cluster
    all_distances = torch.zeros(batch_size, num_classes, device=device)

    # Calculate Mahalanobis distance to each cluster
    for class_id in range(num_classes):
        cluster = stream_clusters.clusters[class_id]

        # Calculate (x - μ) for all embeddings in the batch
        centered = embeddings - cluster.mu

        # Calculate Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        # Using einsum for efficient matrix multiplication
        mahalanobis_distances = torch.einsum('bi,ij,bj->b', centered, cluster.inv_sigma, centered)
        all_distances[:, class_id] = mahalanobis_distances

    # Find minimum distance and corresponding cluster for each embedding
    min_distances, closest_clusters = torch.min(all_distances, dim=1)

    # Create mask for OOD examples (where min distance > threshold)
    ood_mask = min_distances > ood_threshold

    # Assign final cluster IDs (-1 for OOD)
    cluster_assignments = closest_clusters.clone()
    cluster_assignments[ood_mask] = -1

    return cluster_assignments