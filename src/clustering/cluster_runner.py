import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from omegaconf import DictConfig
from src.clustering.streaming_clusterer import StreamingClusterer
from src.clustering.data_stream import DataStreamSimulator
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import os

def run_streaming_experiment(model: nn.Module, dataset: torch.utils.data.Dataset, config: DictConfig):
    """
    Main function to run the streaming clustering experiment.

    Args:
        model: Your pretrained model (ViT or DINOv2)
        dataset: Your CIFAR100Dataset instance
        config: Your existing config
        stream_config: Streaming configuration (uses defaults if None)
    """

    # Create streaming clusterer
    logging.info(f"Initializing {config.clustering.clustering_algorithms.name} streaming clusterer...")
    clusterer = StreamingClusterer(model=model, config=config, device=config.device)

    # Create data stream
    logging.info("Setting up data stream...")
    stream_simulator = DataStreamSimulator(dataset=dataset,config=config)

    # Run streaming experiment
    logging.info("Starting streaming clustering experiment...")
    logging.info(f"Total samples: {stream_simulator.total_samples}")
    logging.info(f"Update interval: {config.clustering.evaluation_interval}")

    print("\n" + "="*60)
    print("STREAMING CLUSTERING EXPERIMENT")
    print("="*60)

    if os.path.exists(f'{config.clustering.clustering_algorithms.name}_extracted_features.pt'):
        dataset = torch.load(f'{config.clustering.clustering_algorithms.name}_extracted_features.pt')
        embedded_normalized_dataloader = DataLoader(
            dataset,
            batch_size=config.clustering.batch_size,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )
    else:
        raw_dataloader = DataLoader(
            dataset, # dataset is a torch.utils.data.TensorDataset
            batch_size=config.clustering.batch_size,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )
        embedded_normalized_dataloader = extract_features(model, raw_dataloader, config.device, config)


    pbar = tqdm(embedded_normalized_dataloader, total=len(embedded_normalized_dataloader), desc="Streaming...")
    try:
        for _, (embedded_features, targets) in enumerate(pbar):
            result = clusterer.process_batch(embedded_features, targets) # Process batch through clustering

            # Update progress bar
            pbar.set_postfix({
                'clusters': result['clusters_discovered'],
                'samples': result['samples_processed']
            })

            # Optional: Add early stopping condition
            if config.clustering.batch_size == 1 and result['samples_processed'] >= 10000:
                logging.info("Reached 10000 samples in one-at-a-time mode, stopping early...")
                break

    except KeyboardInterrupt:
        logging.info("Streaming interrupted by user")
    finally:
        pbar.close()

    # Final evaluation
    logging.info("\nPerforming final evaluation...")
    final_metrics = clusterer._evaluate()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total samples processed: {clusterer.samples_seen}")
    print(f"Final clusters discovered: {clusterer.clusters_discovered}")

    if final_metrics:
        for key, value in final_metrics.items():
            if key not in ['timestamp', 'samples_seen']:
                print(f"{key}: {value:.4f}")

    # Plot results
    plot_streaming_results(clusterer.metrics_history, config.clustering.clustering_algorithms.name)

    return clusterer


def plot_streaming_results(metrics_history: Dict[str, List], algorithm_name: str):
    """Plot the streaming clustering results over time"""

    import matplotlib.pyplot as plt

    if not metrics_history['n_clusters']:
        logging.warning("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Number of clusters over time
    axes[0, 0].plot(metrics_history['samples_seen'],
                    metrics_history['n_clusters'], 'b-')
    axes[0, 0].set_xlabel('Samples Seen')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].set_title('Cluster Discovery Over Time')
    axes[0, 0].grid(True)

    # Silhouette score
    if 'silhouette' in metrics_history and metrics_history['silhouette']:
        axes[0, 1].plot(metrics_history['samples_seen'][:len(metrics_history['silhouette'])],
                       metrics_history['silhouette'], 'g-')
        axes[0, 1].set_xlabel('Samples Seen')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Clustering Quality (Silhouette)')
        axes[0, 1].grid(True)

    # ARI if available
    if 'ari' in metrics_history and metrics_history['ari']:
        axes[1, 0].plot(metrics_history['samples_seen'][:len(metrics_history['ari'])],
                       metrics_history['ari'], 'r-')
        axes[1, 0].set_xlabel('Samples Seen')
        axes[1, 0].set_ylabel('ARI')
        axes[1, 0].set_title('Adjusted Rand Index vs Ground Truth')
        axes[1, 0].grid(True)

    # Processing speed
    if 'timestamp' in metrics_history and metrics_history['timestamp']:
        speeds = [s/t for s, t in zip(metrics_history['samples_seen'],
                                      metrics_history['timestamp'])]
        axes[1, 1].plot(metrics_history['samples_seen'], speeds, 'm-')
        axes[1, 1].set_xlabel('Samples Seen')
        axes[1, 1].set_ylabel('Samples/Second')
        axes[1, 1].set_title('Processing Speed')
        axes[1, 1].grid(True)

    plt.suptitle(f'{algorithm_name.upper()} Streaming Clustering Results')
    plt.tight_layout()
    plt.savefig(f'{algorithm_name}_streaming_results.png')
    logging.info(f"Results saved to {algorithm_name}_streaming_results.png")
    plt.show()

def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device, config: DictConfig) -> DataLoader:
    """
    Extract features from the dataset using the model.

    Args:
        model: The model to use for feature extraction
        dataloader: The dataloader to use for feature extraction
        device: The device to use for feature extraction
    """
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            features = model(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Normalize features
    logging.info("Normalizing features...")
    scaler = StandardScaler()
    all_features = scaler.fit_transform(np.concatenate(all_features).reshape(-1, all_features[0].shape[-1]))
    logging.info(f"Successfully normalized {len(all_features)} features...")

    all_features = torch.from_numpy(all_features)
    all_labels = torch.from_numpy(np.concatenate(all_labels))

    # Create new pytorch dataloader with normalized features and labels
    dataset = torch.utils.data.TensorDataset(all_features, all_labels)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.clustering.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # Save the dataloader
    torch.save(dataset, f'{config.clustering.clustering_algorithms.name}_extracted_features.pt')

    logging.info(f"Successfully created dataloader with {len(dataset)} samples...")
    return dataloader