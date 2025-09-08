from collections import deque
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
import time
import logging
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from river import cluster
from sklearn.mixture import BayesianGaussianMixture
from omegaconf import DictConfig

class StreamingClusterer:
    """
    Online clustering system that processes streaming data.
    Supports multiple clustering algorithms and evaluation metrics.
    """

    def __init__(self, model: nn.Module, config: DictConfig, device: torch.device):

        self.model = model # The feature extractor (e.g. a ViT model)
        self.stream_config = config.clustering # The clustering algorithm configuration
        self.device = device # The device to run the clustering algorithm on

        # Initialize the clustering algorithm
        self.clustering_mechanism = self._init_clustering_algorithm()

        # Buffers for streaming
        self.feature_buffer = deque(maxlen=self.stream_config.buffer_size)
        self.label_buffer = deque(maxlen=self.stream_config.buffer_size)
        self.prediction_buffer = deque(maxlen=self.stream_config.buffer_size)

        # Metrics tracking
        self.metrics_history = {
            'n_clusters': [],
            'silhouette': [],
            'davies_bouldin': [],
            'ari': [],
            'nmi': [],
            'samples_seen': []
        }

        # Streaming state
        self.samples_seen = 0
        self.clusters_discovered = 0
        self.eval_counter = 0

        logging.info(f"Initialized {self.stream_config.clustering_algorithms.name} clustering with batch_size={self.stream_config.batch_size}")

    def _init_clustering_algorithm(self):
        """Initialize the chosen clustering algorithm"""
        algo = self.stream_config.clustering_algorithms.name.lower()

        if algo == 'denstream':
            return cluster.DenStream(
                decaying_factor=self.stream_config.clustering_algorithms.lambda_,
                beta=self.stream_config.clustering_algorithms.beta,
                mu=self.stream_config.clustering_algorithms.mu,
                epsilon=self.stream_config.clustering_algorithms.epsilon,
                n_samples_init=self.stream_config.buffer_size
            )

        elif algo == 'dbstream':
            return cluster.DBSTREAM(
                clustering_threshold=self.stream_config.clustering_algorithms.threshold,
                fading_factor=self.stream_config.clustering_algorithms.fading,
                cleanup_interval=self.stream_config.clustering_algorithms.cleanup_interval,
                intersection_factor=self.stream_config.clustering_algorithms.intersection_factor,
                minimum_weight=self.stream_config.clustering_algorithms.minimum_weight
            )

        elif algo == 'dpgmm':
            return BayesianGaussianMixture(
                n_components=self.stream_config.clustering_algorithms.components,
                weight_concentration_prior_type=self.stream_config.clustering_algorithms.weight_concentration_prior_type,
                weight_concentration_prior=self.stream_config.clustering_algorithms.concentration,
                covariance_type=self.stream_config.clustering_algorithms.covariance_type,
                max_iter=self.stream_config.clustering_algorithms.max_iter,
                warm_start=self.stream_config.clustering_algorithms.warm_start,
                init_params=self.stream_config.clustering_algorithms.init_params
            )

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def process_batch(self, embedded_features: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """
        Process a batch of images through feature extraction and clustering.
        This is the main streaming interface.
        """
        # Add to buffers
        self.feature_buffer.extend(embedded_features)
        self.label_buffer.extend(labels.cpu().numpy())
        self.samples_seen += len(embedded_features)

        # Update clusters based on algorithm type
        predictions = self._update_clusters(embedded_features)

        # Evaluate if at interval
        metrics = None
        # if self.samples_seen % self.stream_config.evaluation_interval == 0:
        #     metrics = self._evaluate()
        #     self._log_progress(metrics)

        return {
            'predictions': predictions,
            'metrics': metrics,
            'samples_processed': self.samples_seen,
            'clusters_discovered': self.clusters_discovered
        }

    def _update_clusters(self, features: np.ndarray) -> np.ndarray:
        """
        Update clustering model with new features.
        """
        predictions = []

        if self.stream_config.clustering_algorithms.name in ['denstream', 'dbstream']:
            # River algorithms - process one by one
            for i, feat in enumerate(features):
                feat_dict = {i: val for i, val in enumerate(feat)}

                # Learn from the sample
                start_time = time.time()
                self.clustering_mechanism.learn_one(feat_dict)
                end_time = time.time()
                print(f"INTERATION {i+self.samples_seen}: Learn one sample time: {round(end_time - start_time, 2)} secs")
                print(self.clustering_mechanism.p_micro_clusters)
                # Get prediction
                # start_time = time.time()
                # pred = self.clustering_mechanism.predict_one(feat_dict)
                # end_time = time.time()
                # print(f"Time taken to predict one sample: {round(end_time - start_time, 2)} seconds")
                # predictions.append(pred if pred is not None else -1)

            # Update cluster count
            unique_clusters = set(predictions) - {-1}
            self.clusters_discovered = len(unique_clusters)

        elif self.stream_config.clustering_algorithms == 'dpgmm':
            # Scikit-learn style - batch update
            if self.samples_seen >= self.stream_config.buffer_size:
                # Use buffer for DPGMM
                buffer_features = np.array(self.feature_buffer)
                self.clustering_mechanism.fit(buffer_features)
                predictions = self.clustering_mechanism.predict(features)

                # Count effective components
                weights = self.clustering_mechanism.weights_
                self.clusters_discovered = np.sum(weights > 0.01)
            else:
                # Not enough samples yet
                predictions = [-1] * len(features)

        self.prediction_buffer.extend(predictions)
        return np.array(predictions)

    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate current clustering quality.
        """
        self.eval_counter += 1
        if len(self.feature_buffer) < 2:
            return {}

        features = np.array(self.feature_buffer)
        labels = np.array(self.label_buffer)
        predictions = np.array(self.prediction_buffer)

        # Filter out noise points
        valid_mask = predictions != -1
        if valid_mask.sum() < 2:
            return {}

        features_valid = features[valid_mask]
        predictions_valid = predictions[valid_mask]
        labels_valid = labels[valid_mask]

        metrics = {
            'n_clusters': self.clusters_discovered,
            'samples_seen': self.samples_seen
        }

        # Clustering quality metrics (no ground truth needed)
        # Only compute expensive metrics every 5 evaluations to save time
        if len(np.unique(predictions_valid)) > 1 and self.eval_counter % 5 == 0:
            try:
                metrics['silhouette'] = silhouette_score(features_valid, predictions_valid)
                metrics['davies_bouldin'] = davies_bouldin_score(features_valid, predictions_valid)
            except Exception:
                pass

        # If we have ground truth labels, compute supervised metrics
        if len(labels_valid) > 0:
            try:
                metrics['ari'] = adjusted_rand_score(labels_valid, predictions_valid)
                metrics['nmi'] = normalized_mutual_info_score(labels_valid, predictions_valid)
            except:
                pass

        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        return metrics

    def _log_progress(self, metrics: Dict[str, float]):
        """Log current progress and metrics"""

        log_msg = f"\n[Stream Progress] Samples: {self.samples_seen} | "
        log_msg += f"Clusters: {self.clusters_discovered}"

        if metrics:
            if 'silhouette' in metrics:
                log_msg += f" | Silhouette: {metrics['silhouette']:.3f}"
            if 'ari' in metrics:
                log_msg += f" | ARI: {metrics['ari']:.3f}"

        logging.info(log_msg)