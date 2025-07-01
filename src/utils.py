import os
import random
import logging
import torch
import numpy as np

from typing import Dict, List, Optional, Tuple
from torch import device as TorchDevice
from torch import version
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import roc_curve, auc

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seeds for reproducibility.
    Args:
        seed (int): The seed value to set.
    Returns:
        None
    """

    random.seed(seed) # Set Python random seed
    np.random.seed(seed)  # Set NumPy random seed
    os.environ["PYTHONHASHSEED"] = str(seed) # Set a fixed value for the hash seed, seeds for data loading operations

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # set torch (GPU) seed
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True # set cudnn to deterministic mode
        torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

    # Document the environment for future reproducibility
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {version.cuda if torch.cuda.is_available() else 'N/A'}")

    logging.info(f"Random seed set to {seed} for reproducibility.")


def get_checkpoint_dict(dataset_name: str, config: DictConfig, device: TorchDevice) -> dict:
    """
    Get the checkpoint dictionary for the given dataset.
    Args:
        dataset_name (str): Name of the dataset
        config (DictConfig): Configuration object
        device (str): Device to load the checkpoint on
    Returns:
        checkpoint_data (dict): Checkpoint dictionary
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state_dict, # Model weights
            'optimizer_state_dict': optimizer_state_dict, # Optimizer state
            'history': history, # Training history
            'class_info': {
                'num_of_classes': num_of_classes, # Old name for this property "n_classes"
                'pretrain_classes': pretrain_class_indicies,
                'left_out_classes': ood_class_indicies, # Old name for this property "continual_classes"
                'left_out_ind_indices': pretrained_left_out_indices, # Old name for this property "left_out_indices"
                'pretrain_class_mapping': pretrain_class_mapping # Old name for this property "class_mapping"
            }
        }
    """
    # Only CIFAR-100 has a separate checkpoint directory
    logging.info(f"Loading checkpoint for {dataset_name} dataset...")
    if dataset_name == 'cifar100':
        checkpoint_path = config.model.backbone.cifar100_location
    elif dataset_name == 'tiny_imagenet':
        checkpoint_path = config.model.backbone.tiny_imagenet_location
    else:
        raise NotImplementedError(f"[Error] Checkpoint data not supported for {dataset_name}")

    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # --- Load Checkpoint ---
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False) # Load directly to target device if possible
        logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    return checkpoint_data


def plot_roc_curves(
    left_out_ind_stats: Dict,
    ood_stats: Dict,
    detector_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "ROC Curves for OOD Detection Methods",
    figsize: Tuple[int, int] = (10, 8),
    show_plot: bool = True
) -> Dict[str, float]:
    """
    Plot ROC curves for multiple OOD detection methods.

    Args:
        left_out_ind_stats (Dict): Dictionary containing OOD detection scores for in-distribution data
        ood_stats (Dict): Dictionary containing OOD detection scores for out-of-distribution data
        detector_names (Optional[List[str]]): List of detector names to plot. If None, plots all available detectors
        save_path (Optional[str]): Path to save the plot. If None, plot is not saved
        title (str): Title for the plot
        figsize (Tuple[int, int]): Figure size for the plot
        show_plot (bool): Whether to display the plot

    Returns:
        Dict[str, float]: Dictionary containing AUROC scores for each detector
    """

    # Available detectors mapping
    detector_mapping = {
        "msp": "all_msp_scores",
        "odin": "all_odin_scores",
        "mahalanobis": "all_mahalanobis_scores",
        "energy": "all_energy_scores",
        "knn": "all_knn_scores"
    }

    # If no specific detectors are specified, use all available ones
    if detector_names is None:
        detector_names = list(detector_mapping.keys())

    # Create figure
    plt.figure(figsize=figsize)

    # Colors for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    aurocs = {}

    for i, detector_name in enumerate(detector_names):
        score_key = detector_mapping.get(detector_name)
        if score_key is None:
            logging.warning(f"Unknown detector: {detector_name}. Skipping.")
            continue

        # Get scores for ID and OOD data
        ind_scores_tensor = left_out_ind_stats.get(score_key)
        ood_scores_tensor = ood_stats.get(score_key)

        if ind_scores_tensor is None or ood_scores_tensor is None:
            logging.warning(f"Scores for '{detector_name}' not found in one or both datasets. Skipping.")
            aurocs[detector_name] = np.nan
            continue

        if not isinstance(ind_scores_tensor, torch.Tensor) or not isinstance(ood_scores_tensor, torch.Tensor):
            logging.warning(f"Scores for '{detector_name}' are not tensors. Skipping.")
            aurocs[detector_name] = np.nan
            continue

        # Convert to numpy
        ind_scores = ind_scores_tensor.cpu().numpy()
        ood_scores = ood_scores_tensor.cpu().numpy()

        # Combine scores and create labels
        all_scores = np.concatenate([ind_scores, ood_scores])
        # Labels: 0 for in-distribution, 1 for out-of-distribution
        all_labels = np.concatenate([
            np.zeros(len(ind_scores), dtype=int),
            np.ones(len(ood_scores), dtype=int)
        ])

        try:
            # For ROC curve, we need to flip scores since our OOD methods typically
            # output higher scores for ID samples (lower scores indicate OOD)
            # Using -all_scores makes higher values correspond to OOD predictions
            fpr, tpr, _ = roc_curve(all_labels, -all_scores)
            roc_auc = auc(fpr, tpr)
            aurocs[detector_name] = roc_auc

            # Plot ROC curve
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{detector_name.upper()} (AUROC = {roc_auc:.3f})')

        except ValueError as e:
            logging.error(f"ROC calculation failed for {detector_name}: {e}")
            aurocs[detector_name] = np.nan

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add text box with summary statistics
    valid_aurocs = {k: v for k, v in aurocs.items() if not np.isnan(v)}
    if valid_aurocs:
        best_method = max(valid_aurocs, key=lambda k: valid_aurocs[k])
        textstr = f'Best Method: {best_method.upper()}\nAUROC: {valid_aurocs[best_method]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curves saved to {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    return aurocs


def plot_single_roc_curve(
    ind_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str = "OOD Detection",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show_plot: bool = True
) -> float:
    """
    Plot ROC curve for a single OOD detection method.

    Args:
        ind_scores (np.ndarray): Scores for in-distribution samples
        ood_scores (np.ndarray): Scores for out-of-distribution samples
        method_name (str): Name of the detection method
        save_path (Optional[str]): Path to save the plot
        figsize (Tuple[int, int]): Figure size
        show_plot (bool): Whether to display the plot

    Returns:
        float: AUROC score
    """

    # Combine scores and create labels
    all_scores = np.concatenate([ind_scores, ood_scores])
    all_labels = np.concatenate([
        np.zeros(len(ind_scores), dtype=int),
        np.ones(len(ood_scores), dtype=int)
    ])

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, -all_scores)
    roc_auc = auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'{method_name} (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {method_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curve saved to {save_path}")

    if show_plot:
        plt.show()

    return float(roc_auc)
