import numpy as np
from sklearn.cluster import Birch
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OnlineClustering:
    """
    A wrapper class for an online unsupervised clustering algorithm that automatically
    determines the number of clusters. It uses the BIRCH algorithm from scikit-learn.

    The class is designed to take an encoder model (like DINOv2) to first generate
    embeddings from input data and then cluster those embeddings without a predefined
    number of clusters. The number of clusters found is controlled by the `threshold` parameter.
    """
    def __init__(self, model: nn.Module, dataloader: DataLoader, embeddings: np.ndarray, branching_factor=50, threshold=1.5):
        """
        Initializes the OnlineClustering instance.

        Args:
            branching_factor (int): The maximum number of CF Subclusters in each node of the CF-Tree.
            threshold (float): The radius of the subcluster obtained by merging a new sample and the
                               closest subcluster. If the radius is larger than the threshold, a new
                               subcluster is created. This parameter is crucial for controlling the
                               final number of clusters. A smaller threshold will result in more clusters.
            model: The neural network model (e.g., DINOv2 ViT) to generate embeddings.
            dataloader (DataLoader): DataLoader providing batches of input data.
        """

        self.clustering_algorithm = Birch(
            n_clusters=None,
            branching_factor=branching_factor,
            threshold=threshold
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.dataloader = dataloader
        # self.embeddings = self._get_embeddings(dataloader)
        self.embeddings = embeddings

        logging.info(f"BIRCH algorithm initialized (on device: '{self.device}')...")
        logging.info(f"BIRCH hyperparameters: branching_factor={branching_factor}, threshold={threshold}.")

    @property
    def n_clusters_found(self) -> int:
        """
        Returns the number of clusters found so far.

        This is only available after the model has been fitted with at least one batch of data.
        The number of clusters corresponds to the number of leaf subclusters in the CF-Tree.

        Returns:
            int: The current number of clusters, or 0 if the model is not fitted.
        """

        if hasattr(self.clustering_algorithm, 'subcluster_labels_'):
            # The labels assigned are the indices of the leaf subclusters
            return self.clustering_algorithm.subcluster_labels_.max() + 1

        return 0

    def _get_embeddings(self, dataloader: DataLoader) -> np.ndarray:
        """
        Extracts all feature embeddings for a given dataset using the provided encoder model.

        Args:
            dataloader (DataLoader): DataLoader providing batches of input data. Directly passing in dataloader to avoid
                                     class creation errors.
        Returns:
            all_embeddings (np.ndarray): A numpy array containing the extracted embeddings for the entire dataset.

        """
        self.model.to(self.device)
        self.model.eval()

        all_embeddings = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Generating Embeddings.."):
                images = images.to(self.device)
                # DINOv2 may return a dictionary, we're interested in the CLS token embeddings
                output = self.model(images)
                embeddings = output.cpu().numpy()
                all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)

    def update_clusters(self, embeddings_batch: np.ndarray) -> np.ndarray:
        """
        Performs an online learning step by partially fitting the BIRCH model on a new
        batch of data. This updates the model's internal CF-Tree.

        This method corresponds to both the "initial learn" and "update" phases of your request.

        Args:
            embeddings_batch (np.ndarray): The current batch of embeddings.
        """

        self.clustering_algorithm.partial_fit(embeddings_batch)


    def predict(self, embeddings_batch: np.ndarray) -> np.ndarray:
        """
        Predicts cluster labels for a new batch of data using the current state of the
        clustering model, without updating it. The predicted label is the index of the
        closest leaf subcluster in the CF-Tree.

        Args:
            embeddings_batch (np.ndarray): The current batch of embeddings.
        Returns:
            predictions (np.ndarray): The array of predicted cluster labels.

        Raises:
            RuntimeError: If the model has not been fitted with at least one batch of data.
        """
        if not hasattr(self.clustering_algorithm, 'root_'):
            raise RuntimeError("Model has not been fitted yet. Call update_clusters() at least once before predicting.")

        predictions = self.clustering_algorithm.predict(embeddings_batch)
        return predictions


    def run_online_simulation(self, model: nn.Module, stream_batch_size: int = 100) -> np.ndarray:
        """
        Runs a full online clustering simulation.

        (1.) Fetch all computed embeddings.
        (2.) Use the first batch for initial learning only.
        Processes embeddings in batches to simulate a data stream, following a 'learn, then predict' cycle.

        Args:
            model (nn.Module): The model to generate embeddings.
            dataloader (DataLoader): DataLoader for the ENTIRE dataset.
            stream_batch_size (int): The number of samples to process in each step of the stream.

        Returns:
            np.ndarray: An array containing the predicted cluster label for every sample in the dataset.
        """
        # Step 1: Get all embeddings
        all_embeddings = self.embeddings
        num_total_samples = len(all_embeddings)
        logging.info(f"Starting online simulation on {num_total_samples} embeddings...")

        all_predictions = np.zeros(num_total_samples, dtype=int)

        # Step 2: Initial batch for initial cluster formation
        first_batch_embeddings = all_embeddings[0:stream_batch_size]
        self.update_clusters(first_batch_embeddings)

        # The labels for the first batch are assigned during the fit
        all_predictions[0:stream_batch_size] = self.clustering_algorithm.labels_
        logging.info(f"Initial learning complete. Found {self.n_clusters_found} clusters...")


        # Step 3: Process subsequent batches of embeddings
        pbar = tqdm(range(stream_batch_size, num_total_samples, stream_batch_size), desc="Processing Stream...")
        for i in pbar:
            start_idx = i
            end_idx = min(i + stream_batch_size, num_total_samples)
            current_batch_embeddings = all_embeddings[start_idx:end_idx]

            if len(current_batch_embeddings) == 0:
                continue

            # Predict based on the model's current state (learned from previous batches)
            predictions = self.predict(current_batch_embeddings)
            all_predictions[start_idx:end_idx] = predictions

            # Now, update the model with this new batch
            self.update_clusters(current_batch_embeddings)

        logging.info(f"====================== ONLINE SIMULATION FINISHED ======================")
        logging.info(f"Final number of clusters: {self.n_clusters_found}")
        logging.info(f"==================================================================")
        return all_predictions



    # =============================== NOT CURRENTLY USED =============================== #
    # Legacy methods which accept the full dataset dataloader vs. a batch of the full dataset.
    def update_clusters_from_dataloader(self, encoder, dataloader):
        """Convenience wrapper to update clusters directly from a dataloader."""
        embeddings = self._get_embeddings(encoder, dataloader)
        self.update_clusters(embeddings)
        logging.info(f"Cluster update complete. Found {self.n_clusters_found} clusters so far.")

    def predict_from_dataloader(self, encoder, dataloader):
        """Convenience wrapper to predict clusters directly from a dataloader."""
        embeddings = self._get_embeddings(encoder, dataloader)
        return self.predict(embeddings)