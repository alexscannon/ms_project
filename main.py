import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from dotenv import load_dotenv

from src.models.backbone.create_model import create_model
from src.data.dataset_loader import dataload
from src.utils import get_checkpoint_dict
from src.models.ood.ood_detector import OODDetector
from src.models.continual_learning.CL import ContinualLearning
from src.loggers.wandb_logger import WandBLogger
from src.utils import set_seed
from src.data.cifar100 import CIFAR100Dataset
from src.clustering.cluster_runner import run_streaming_experiment
from src.clustering_2.clustering_2 import OnlineClustering
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.data.utils import get_embeddings

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # logging.info(f"Loaded Configuration: {OmegaConf.to_yaml(config)}")
    load_dotenv()
    logging.info(f"Loaded Configuration...")

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
    # Create new non-pretrained ViT model
    model = create_model(
        img_size=config.data.image_size,
        n_classes=int(config.data.num_classes * config.ind_class_ratio),
        config=config
    )

    # load model weights if there exists pretrained weights.
    if config.model.pretrained and config.model.backbone.name == 'vit':
        logging.info("Loading pre-trained Vision Transformer model...")
        checkpoint_data = get_checkpoint_dict(config.data.name, config, device)

        logging.info(f"checkpoint_data attributes: {checkpoint_data.keys()}")
        logging.info(f"checkpoint_data['class_info'] attributes: {list(checkpoint_data['class_info'].keys())}")
        model.load_state_dict(checkpoint_data["model_state_dict"])

    # ======================================================== #
    # ==================== OLD EXPERIMENT ==================== #
    # ======================================================== #
    if config.is_old_experiment:
        # ============================ Dataset Loading ============================ #
        # Load remaining ID and OOD datasets
        logging.info("Loading left-out IND, OOD, and Corrupted datasets...")
        left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader, corrupted_dataloader = dataload(config, checkpoint_data)

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
        continual_learning = ContinualLearning(
            config=config, model=model, device=device, left_in_dataloader=left_in_ind_dataloader
        )
        continual_learning.run_covariate_continual_learning_inference(
            left_out_ind_dataloader=left_out_ind_dataloader,
            ood_dataloader=ood_dataloader,
            wandb_logger=wand_logger,
            ood_detector=ood_detector,
            config=config,
            model=model,
            checkpoint_class_info=checkpoint_data["class_info"],
            corrupted_dataloader=corrupted_dataloader
        )
    # ======================================================== #
    # ==================== NEW EXPERIMENT ==================== #
    # ======================================================== #
    else:
        logging.info(f"Loading dataset: {config.data.name}...")
        dataset = CIFAR100Dataset(config).all_data
        logging.info(f"Successfully loaded dataset...")

        # run_streaming_experiment(model, dataset, config)

        full_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.data.batch_size,
            shuffle=config.data.shuffle # Shuffle is False to simulate a consistent stream
        )
        best_threshold, best_cluster_diff = 0, float('inf')
        best_ari, best_nmi, best_ari_threshold, best_nmi_threshold = 0, 0, 0, 0

        embeddings, true_labels = get_embeddings(dataloader=full_dataloader, model=model, device=device)

        for t in range(4500, 4550, 1):
            print(f"======================================================================")
            print(f"======================= Threshold: {t} ===============================")
            print(f"======================================================================")
            threshold = t/100

            online_clusterer = OnlineClustering(
                model=model,
                dataloader=full_dataloader,
                embeddings=embeddings,
                true_labels=true_labels,
                threshold=threshold, # This value requires tuning!
                branching_factor=50,
            )

            # Entire simulation: (1.) Uses the pre-computed embeddings (2.) learns from the first batch,
            # (3.) iterates through the rest, predicting then updating clusters.
            final_predictions, final_metrics = online_clusterer.run_online_simulation(
                stream_batch_size=100
            )

            # print(f"Final metrics: {final_metrics}")

            n_clusters_found = online_clusterer.n_clusters_found
            true_cluster_diff = abs(n_clusters_found - config.data.num_classes)

            if true_cluster_diff < best_cluster_diff:
                best_cluster_diff = true_cluster_diff
                best_threshold = threshold

            final_ari = final_metrics['final_ari']
            final_nmi = final_metrics['final_nmi']

            if final_ari > best_ari:
                best_ari = final_ari
                best_ari_threshold = threshold

            if final_nmi > best_nmi:
                best_nmi = final_nmi
                best_nmi_threshold = threshold

            # print("\n======================= Simulation Summary =======================")
            # print(f"Total samples processed: {len(final_predictions)}")

        print(f"Final number of clusters discovered: {n_clusters_found}")
        print(f"BEST CLUSTER THRESHOLD FOUND: threshold: {best_threshold}, best cluster diff: +/-{best_cluster_diff}")
        print(f"Best ARI {best_ari} with threshold: {best_ari_threshold}")
        print(f"Best NMI {best_nmi} with threshold: {best_nmi_threshold}")
        print(f"======================================================================")


    wand_logger.finish(exit_code=0)


if __name__ == "__main__":
    main()