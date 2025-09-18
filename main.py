import hydra
from omegaconf import DictConfig
import logging
from src.models.backbone.create_model import create_model
from src.data.dataset_loader import dataload
from src.utils import get_checkpoint_dict, get_device
from src.models.ood.ood_detector import OODDetector
from src.models.continual_learning.CL import ContinualLearning
from src.loggers.wandb_logger import WandBLogger
from src.utils import setup_experiment
from src.data.cifar100 import CIFAR100Dataset
from src.clustering.cluster_runner import run_streaming_experiment
from src.clustering_2.clustering_2 import OnlineClustering
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.utils import load_embeddings
import time
from src.utils import visualize_clusters

logger = logging.getLogger("msproject")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # ============================ Experiment setup ============================ #
    # Load config + set up logger + set random seed (reproducibility)
    setup_experiment(config)

    # Determine and set experiment's device
    device = get_device(config)

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
        logger.info("Loading pre-trained Vision Transformer model...")
        checkpoint_data = get_checkpoint_dict(config.data.name, config, device)

        logger.info(f"checkpoint_data attributes: {checkpoint_data.keys()}")
        logger.info(f"checkpoint_data['class_info'] attributes: {list(checkpoint_data['class_info'].keys())}")
        model.load_state_dict(checkpoint_data["model_state_dict"])

    # ======================================================== #
    # ==================== OLD EXPERIMENT ==================== #
    # ======================================================== #
    if config.is_old_experiment:
        # ============================ Dataset Loading ============================ #
        # Load remaining ID and OOD datasets
        logger.info("Loading left-out IND, OOD, and Corrupted datasets...")
        left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader, corrupted_dataloader = dataload(config, checkpoint_data)

        # ============================ OOD detection ============================ #
        # Create OOD detector
        logger.info("Creating OOD detector...")
        ood_detector = OODDetector(config, model, left_in_ind_dataloader, device)

        # Run OOD detection
        # logger.info("Running OOD detection...")
        # aurocs = ood_detector.run_ood_detection(left_in_ind_dataloader, left_out_ind_dataloader, ood_dataloader)
        # formatted_aurocs = {k: f"{v * 100:.2f}%" for k, v in aurocs.items()}
        # logger.info(f"AUROCs: {formatted_aurocs}")

        # ============================ Continual Learning ============================ #
        logger.info("Running Continual Learning scenario...")
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

    # ==================== NEW EXPERIMENT ==================== #
    else:
        logger.info(f"Loading dataset: {config.data.name}...")
        dataset = CIFAR100Dataset(config).all_data
        logger.info(f"Successfully loaded dataset...")

        # run_streaming_experiment(model, dataset, config)

        full_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.data.batch_size,
            shuffle=config.data.shuffle # Shuffle is False to simulate a consistent stream
        )

        embeddings, true_labels = load_embeddings(config, model, device, full_dataloader)

        best_threshold, best_cluster_diff = 0, float('inf')
        best_ari, best_nmi, best_ari_threshold, best_nmi_threshold = 0, 0, 0, 0
        pbar = tqdm(range(2530, 3000, 1), desc="Processing Stream...")
        for idx, t in enumerate(pbar):
            start_time = time.time()
            threshold = t / 100
            # threshold = t / 100 or something

            logger.info(f"======================================================================")
            logger.info(f"======================= Threshold: {threshold} ===============================")
            logger.info(f"======================================================================")

            online_clusterer = OnlineClustering(
                model=model,
                dataloader=full_dataloader,
                embeddings=embeddings,
                true_labels=true_labels,
                threshold=threshold,
                branching_factor=50,
            )

            # Entire simulation: (1.) Uses the pre-computed embeddings (2.) learns from the first batch,
            # (3.) iterates through the rest, predicting then updating clusters.
            final_predictions, final_metrics = online_clusterer.run_online_simulation(
                stream_batch_size=100
            )

            n_clusters_found = online_clusterer.n_clusters_found
            true_cluster_diff = abs(n_clusters_found - config.data.num_classes)

            final_ari = final_metrics['final_ari']
            final_nmi = final_metrics['final_nmi']

            if true_cluster_diff < best_cluster_diff:
                best_cluster_diff_labels = final_predictions
                logger.info(f"NEW BEST CLUSTER DIFF – diff +/-{true_cluster_diff} (Threshold: {threshold})")
                best_cluster_diff = true_cluster_diff
                best_threshold = threshold

            if final_ari > best_ari:
                logger.info(f"NEW BEST ARI – {final_ari} (Threshold: {threshold})")
                best_ari_labels = final_predictions
                best_ari = final_ari
                best_ari_threshold = threshold

            if final_nmi > best_nmi:
                logger.info(f"NEW BEST NMI – {final_nmi} (Threshold: {threshold})")
                best_nmi_labels = final_predictions
                best_nmi = final_nmi
                best_nmi_threshold = threshold

            end_time = time.time()
            logger.info(f"Run #{idx} duration: {end_time - start_time} seconds")

            # logger.info("\n======================= Simulation Summary =======================")
            # logger.info(f"Total samples processed: {len(final_predictions)}")

        logger.info(f"======================================================================")
        # logger.info(f"Final number of clusters discovered: {n_clusters_found}")
        logger.info(f"Best cluster diff: +/-{best_cluster_diff}. Threshold: {best_threshold}")
        logger.info(f"Best ARI {best_ari}. Threshold: {best_ari_threshold}")
        logger.info(f"Best NMI {best_nmi}. Threshold: {best_nmi_threshold}")
        logger.info(f"======================================================================")
        visualize_clusters(
            embeddings,
            best_cluster_diff_labels,
            save_path="visuals/best_cluster/birch_clustering_results.png",
            title=f"BIRCH Clustering (Best Cluster Diff)\nThreshold: {best_threshold}",
            show_plot=True # Set to True if you want to see the plot interactively
        )
        visualize_clusters(
            embeddings,
            best_ari_labels,
            save_path="visuals/best_ari/birch_clustering_results.png",
            title=f"BIRCH Clustering (Best ARI)\nThreshold: {best_ari_threshold}",
            show_plot=True # Set to True if you want to see the plot interactively
        )
        visualize_clusters(
            embeddings,
            best_nmi_labels,
            save_path="visuals/best_nmi/birch_clustering_results.png",
            title=f"BIRCH Clustering (Best NMI)\nThreshold: {best_nmi_threshold}",
            show_plot=True # Set to True if you want to see the plot interactively
        )
    wand_logger.finish(exit_code=0)


if __name__ == "__main__":
    main()