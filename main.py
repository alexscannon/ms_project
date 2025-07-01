import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging

from src.models.backbone.create_model import create_model
from src.data.dataset_loader import dataload
from src.utils import get_checkpoint_dict
from src.models.ood.ood_detector import OODDetector
from src.models.continual_learning.CL import ContinualLearning
from src.loggers.wandb_logger import WandBLogger
from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    logging.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    # ============================ Experiment setup ============================ #
    # Set random seed for reproducibility
    set_seed(config.seed)

    # Determine device
    if torch.cuda.is_available() and hasattr(config, 'device') and config.device == "gpu":
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device} ...")

    # Initialize logging
    wand_logger = WandBLogger(config)


    # ============================ Load pre-trained model ============================ #
    logging.info("Loading pre-trained Vision Transformer model...")
    checkpoint_data = get_checkpoint_dict(config.data.name, config, device)

    logging.info(f"checkpoint_data attributes: {checkpoint_data.keys()}")
    logging.info(f"checkpoint_data['class_info'] attributes: {list(checkpoint_data['class_info'].keys())}")

    # Create new non-pretrained ViT model
    model = create_model(config.data.image_size, int(config.data.num_classes * config.ind_class_ratio), config)
    # load model weights if pretrained in some way
    if config.model.pretrained:
        model.load_state_dict(checkpoint_data["model_state_dict"])

    # ============================ Dataset Loading ============================ #
    # Load remaining ID and OOD datasets
    logging.info("Loading remaining ID and the OOD datasets...")
    left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader = dataload(config, checkpoint_data)

    # ============================ OOD detection ============================ #
    # Create OOD detector
    logging.info("Creating OOD detector...")
    ood_detector = OODDetector(config, model, left_in_ind_dataloader, device)

    # Run OOD detection
    # logging.info("Running OOD detection...")
    # aurocs = ood_detector.run_ood_detection(left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader)
    # formatted_aurocs = {k: f"{v * 100:.2f}%" for k, v in aurocs.items()}
    # logging.info(f"AUROCs: {formatted_aurocs}")

    # ============================ Continual Learning ============================ #
    logging.info("Running Continual Learning scenario...")
    continual_learning = ContinualLearning(config, model, device)
    # Stage 1: Handle the remaining in-distribution data in a continual learning setting
    continual_learning.run_covariate_continual_learning(
        left_out_ind_dataloader=left_in_ind_dataloader,
        ood_dataloader=ood_dataloader,
        wandb_logger=wand_logger,
        ood_detector=ood_detector,
        config=config,
        model=model,
        checkpoint_class_info=checkpoint_data["class_info"]
    )

    wand_logger.finish(exit_code=0)


if __name__ == "__main__":
    main()