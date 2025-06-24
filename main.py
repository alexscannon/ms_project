import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
import os

from src.models.backbone.vit import VisionTransformer
from src.models.backbone.create_model import create_model
from src.data.dataset_loader import dataload
from src.utils import get_checkpoint_dict
from src.models.ood.ood_detector import OODDetector

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    logging.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    # ============================ Experiment setup ============================ #
    # Determine device
    if torch.cuda.is_available() and hasattr(config, 'device') and config.device == "gpu":
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device} ...")


    # ============================ Load pre-trained model ============================ #
    logging.info("Loading pre-trained Vision Transformer model...")
    checkpoint_data = get_checkpoint_dict(config.data.name, config, device)

    logging.info(f"checkpoint_data attributes: {checkpoint_data.keys()}")
    if config.data.name != 'tiny_imagenet':
        logging.info(f"checkpoint_data['class_info'] attributes: {list(checkpoint_data['class_info'].keys())}")

    # Create bare ViT model and load model weights
    model = create_model(config.data.image_size, int(config.data.num_classes * config.ind_class_ratio), config)
    if config.model.pretrained:
        model.load_state_dict(checkpoint_data["model_state_dict"])

    # ============================ Dataset Loading ============================ #
    # Load remaining ID and OOD datasets
    logging.info("Loading remaining ID and the OOD datasets...")
    left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader = dataload(config, checkpoint_data)

    # ============================ OOD detection ============================ #
    # Create OOD detector
    logging.info("Creating OOD detector...")
    ood_detector = OODDetector(config, model, device)

    # Run OOD detection
    logging.info("Running OOD detection...")
    aurocs = ood_detector.run_ood_detection(left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader)
    formatted_aurocs = {k: f"{v * 100:.2f}%" for k, v in aurocs.items()}
    logging.info(f"AUROCs: {formatted_aurocs}")


if __name__ == "__main__":
    main()