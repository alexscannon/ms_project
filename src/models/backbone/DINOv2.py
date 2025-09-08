import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ============================================
# Basic DINOv2 Setup and Feature Extraction
# ============================================

def load_dinov2_model(model_size: str, device: torch.device):
    """
    Load a DINOv2 model from torch hub.

    Available sizes:
    - 'small': ViT-S/14 (21M parameters)
    - 'base': ViT-B/14 (86M parameters)
    - 'large': ViT-L/14 (300M parameters)
    - 'giant': ViT-g/14 (1.1B parameters)
    """
    # Map friendly names to actual model names
    model_map = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14',
        'large': 'dinov2_vitl14',
        'giant': 'dinov2_vitg14'
    }

    model_name = model_map.get(model_size, 'dinov2_vitb14')

    # Load the model from Facebook Research's repository
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.to(device)

    # Set to evaluation mode (important for batch norm and dropout)
    model.eval()

    return model