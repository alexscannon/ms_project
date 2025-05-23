import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.models.backbone.vit import VisionTransformer
import logging

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    logging.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    # Determine device
    if torch.cuda.is_available() and hasattr(config, 'system') and hasattr(config.system, 'device') and 'cuda' in config.system.device:
        device = torch.device(config.system.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load pre-trained model
    logging.info("Loading pre-trained Vision Transformer model...")
    model = VisionTransformer(config=config)



if __name__ == "__main__":
    main()