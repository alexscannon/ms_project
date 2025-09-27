import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import logging
from collections import defaultdict

logger = logging.getLogger("msproject")

class ClusteringEvaluator:
    """
    A class to handle the evaluation of a clustering model over time.
    It stores the history of the evaluation metrics.
    """

    def __init__(self, clustering_algorithm):
        self.metrics_history = {
            'samples_processed': [],
            'ari': [],
            'nmi': [],
            'n_clusters': [],
            'running_ari': [],
            'running_nmi': [],
            'running_pred_cluster_counts': [],
            'running_true_cluster_counts': []
        }
        self.clustering_algorithm = clustering_algorithm
        self.running_embeddings = []
        self.running_true_labels = []
        self.running_predictions = []

        self.clusters = defaultdict(list)

        logger.info("ClusteringEvaluator initialized.")

    def evaluate_and_log(self, curr_true_labels: np.ndarray, curr_predictions: np.ndarray, n_clusters: int, samples_processed: int):
        """
        Calculates clustering metrics for the current state and logs them.

        This method should be called after each batch in the online simulation.
        It evaluates the performance on all data seen up to the current point.

        Args:
            curr_true_labels (np.ndarray): Ground truth labels for all samples processed so far.
            curr_predictions (np.ndarray): Predicted cluster labels for all samples processed so far.
            n_clusters (int): The current number of clusters discovered by the model.
            samples_processed (int): The total number of samples processed.
        """
        if len(curr_true_labels) != len(curr_predictions):
            logger.error("Mismatch between number of true labels and predictions. Cannot evaluate.")
            return

        self.running_predictions.extend(curr_predictions)
        self.running_true_labels.extend(curr_true_labels)

        # for label, prediction in zip(true_labels, predictions):
        #     self.clusters[prediction].append(label)
        for c in np.unique(curr_predictions):
            self.clusters[c].extend(curr_true_labels[curr_predictions == c].tolist())

        num_running_samples = len(self.running_true_labels)
        num_unique_labels = len(np.unique(self.running_true_labels))

        if num_running_samples < 2 * num_unique_labels or num_unique_labels < 2:
            # Not enough data for a meaningful evaluation, so we skip it.
            # We'll still record the number of clusters and samples processed.
            ari_score = 0.0
            nmi_score = 0.0
        else:
            ari_score = adjusted_rand_score(self.running_true_labels, self.running_predictions)
            nmi_score = normalized_mutual_info_score(self.running_true_labels, self.running_predictions)

        self.metrics_history['samples_processed'].append(samples_processed)
        self.metrics_history['ari'].append(ari_score)
        self.metrics_history['nmi'].append(nmi_score)
        self.metrics_history['n_clusters'].append(n_clusters)


    def get_final_metrics(self):
        """
        Returns the final computed metrics.
        """

        return {
            'final_ari': self.metrics_history['ari'][-1],
            'final_nmi': self.metrics_history['nmi'][-1],
            'final_n_clusters': self.metrics_history['n_clusters'][-1],
            'running_ari': self.metrics_history['running_ari'],
            'running_nmi': self.metrics_history['running_nmi'],
            'running_pred_cluster_counts': self.metrics_history['running_pred_cluster_counts'],
            'running_true_cluster_counts': self.metrics_history['running_true_cluster_counts']
        }

    def print_final_summary(self):
        """
        Prints a summary of the final evaluation metrics.
        """
        final_metrics = self.get_final_metrics()
        if final_metrics:
            logger.info("--------- Final Clustering Evaluation ---------")
            logger.info(f"Adjusted Rand Index (ARI): {final_metrics['final_ari']:.4f}")
            logger.info(f"Normalized Mutual Information (NMI): {final_metrics['final_nmi']:.4f}")
            logger.info(f"Final number of clusters discovered: {final_metrics['final_n_clusters']}\n")
        else:
            logger.info("No evaluation metrics were recorded.")

    def evaluate_running_metrics(self, current_embeddings: np.ndarray, n_clusters: int):
        """
        Re-predicts the clusters with the current model state.
        Args:
            current_embeddings (np.ndarray): The current embeddings to predict.
        """
        self.running_embeddings.extend(current_embeddings)
        cuurent_total_predictions = self.clustering_algorithm.predict(self.running_embeddings)

        self.metrics_history['running_ari'].append(adjusted_rand_score(self.running_true_labels, cuurent_total_predictions))
        self.metrics_history['running_nmi'].append(normalized_mutual_info_score(self.running_true_labels, cuurent_total_predictions))
        self.metrics_history['running_pred_cluster_counts'].append(n_clusters)
        self.metrics_history['running_true_cluster_counts'].append(len(set(self.running_true_labels)))
