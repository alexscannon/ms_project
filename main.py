import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.models.backbone.vit import VisionTransformer
import logging

from src.data.dataset_loader import create_datasets
from src.utils import get_checkpoint_dict
from src.models.ood.ood_detector import OODDetector

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    logging.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    # ============================ Experiment setup ============================ #
    # Determine device
    if torch.cuda.is_available() and hasattr(config, 'system') and hasattr(config.system, 'device') and 'cuda' in config.system.device:
        device = torch.device(config.system.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")


    # ============================ Load pre-trained model ============================ #
    logging.info("Loading pre-trained Vision Transformer model...")
    checkpoint_data = get_checkpoint_dict(config.data.name, config, device)
    logging.info(f"checkpoint_data attributes: {checkpoint_data.keys()}")
    logging.info(f"checkpoint_data['class_info'] attributes: {checkpoint_data['class_info'].keys()}")
    model = VisionTransformer(config=config, checkpoint_data=checkpoint_data)


    # ============================ OOD detection ============================ #
    # Load remaining ID and OOD datasets
    logging.info("Loading remaining ID and the OOD datasets...")
    left_out_ind_dataloader, ood_dataloader, pretrained_ind_dataloader = create_datasets(config, checkpoint_data)

    # Create OOD detector
    logging.info("Creating OOD detector...")
    ood_detector = OODDetector(config, model, device)

    # Run OOD detection
    logging.info("Running OOD detection...")
    aurocs = ood_detector.run_ood_detection(left_out_ind_dataloader, ood_dataloader, pretrained_ind_dataloader)
    formatted_aurocs = {k: f"{v * 100:.2f}%" for k, v in aurocs.items()}
    logging.info(f"AUROCs: {formatted_aurocs}")


if __name__ == "__main__":
    main()