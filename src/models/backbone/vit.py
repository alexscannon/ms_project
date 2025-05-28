import os
from typing import Dict
import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm import create_model
import logging # Import logging
from src.utils import get_checkpoint_dict

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisionTransformer(nn.Module):
    def __init__(self, config: DictConfig, checkpoint_data: Dict):
        super().__init__()

        self.config = config
        self.dataset_name = config.data.name
        self.device = config.device
        self.model_name = self.config.model.backbone.get('name', 'vit_small_patch16_224')
        self.checkpoint_data = checkpoint_data

        self.model = self.load_pretrained_model() # Renamed 'location' to 'checkpoint_type' for clarity if it means 'best', 'last' etc.

    def load_pretrained_model(self) -> nn.Module:
        """
        Load a pretrained ViT model from the location specified in the config.
        """
        # Load the checkpoint data
        model_state_dict_key = self.config.model.backbone.checkpoint_keys.get('model_state_dict', 'model_state_dict')

        # Load the model state dictionary
        state_dict = self.checkpoint_data[model_state_dict_key]

        # --- Model Creation & Loading ---
        try:
            model = self.create_raw_vit(num_classes=self.config.model.backbone.num_classes)
            if not self.config.model.pretrained:
                return model # Do not load pretrained weights if user wants raw ViT

            model.load_state_dict(state_dict) # load the pre-trained ViT parameters
            model.to(self.device) # Ensure model is on the correct device

            logging.info(f"Loaded {self.dataset_name} ViT model with {self.config.model.backbone.num_classes} number of classes ...")
            return model # Return the loaded model
        except RuntimeError as e:
            logging.error(f"Failed to load state dict into model: {e}")
            raise RuntimeError(f"Failed to load state dict into model: {e}")


    def create_raw_vit(self, num_classes: int) -> nn.Module:
        # If num_classes=0, this removes classification head, allowing model to be used as a feature extractor
        model = create_model(
            model_name=self.model_name,
            pretrained=False,
            num_classes=num_classes,
            patch_size=self.config.model.backbone.patch_size,
            drop_path_rate=self.config.model.backbone.dropout_rate,
        )
        logging.info(f"Created raw ViT model '{self.model_name}' with {num_classes} output classes.")
        return model




