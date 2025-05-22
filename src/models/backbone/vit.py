import os
import torch
import torch.nn as nn
from omegaconf import DictConfig
from timm import create_model
import logging # Import logging


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisionTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.dataset_name = config.data.name
        self.device = config.device
        self.model = self.load_pretrained_model() # Renamed 'location' to 'checkpoint_type' for clarity if it means 'best', 'last' etc.

    def load_pretrained_model(self) -> nn.Module:
        """
        Load a pretrained ViT model from the location specified in the config.
        """
        # --- Path Configuration --- #
        # Only CIFAR-100 has a separate checkpoint directory
        if self.dataset_name == 'cifar100':
            checkpoint_path = os.path.join(self.config.model.backbone.location, self.dataset_name, self.config.model.backbone.model_filename)
        else:
            checkpoint_path = os.path.join(self.config.model.backbone.location, self.config.model.backbone.model_filename)

        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # --- Load Checkpoint ---
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device) # Load directly to target device if possible
            logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")

        # --- Checkpoint Structure ---
        # TODO: Logic is specific to how this ViT was pretrained (see pretrain directory)
        # Consider making these keys configurable via self.config.model.checkpoint_keys
        model_state_dict_key = self.config.model.backbone.checkpoint_keys.get('model_state_dict', 'model_state_dict')

        try:
            state_dict = checkpoint_data[model_state_dict_key]
        except KeyError as e:
            logging.error(f"Missing expected key in checkpoint {checkpoint_path}: {e}")
            raise KeyError(f"Missing expected key in checkpoint {checkpoint_path}: {e}")


        # --- Model Creation & Loading ---
        try:
            model = self.create_raw_vit(num_classes=self.config.model.backbone.num_classes)
            if not self.config.model.pretrained:
                return model # Do not load pretrained weights if user wants raw ViT

            model.load_state_dict(state_dict) # load the pre-trained ViT parameters
            model.to(self.device) # Ensure model is on the correct device

            logging.info(f"Loaded {self.dataset_name} ViT model with {self.config.model.backbone.num_classes} classes from {checkpoint_path}")
            return model # Return the loaded model
        except RuntimeError as e:
            logging.error(f"Failed to load state dict into model: {e}")
            raise RuntimeError(f"Failed to load state dict into model: {e}")


    def create_raw_vit(self, num_classes: int) -> nn.Module:
        # If num_classes=0, this removes classification head, allowing model to be used as a feature extractor
        model_name = self.config.model.backbone.get('name', 'vit_small_patch16_224')
        model = create_model(model_name, pretrained=False, num_classes=num_classes)

        logging.info(f"Created raw ViT model '{model_name}' with {num_classes} output classes.")
        return model




