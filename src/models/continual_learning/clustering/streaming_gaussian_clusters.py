from torch import device as TorchDevice
from torch.utils.data import DataLoader

from gaussian_cluster import GaussianCluster

class StreamingGaussianClusters:
    def __init__(self, num_classes: int, training_dataloader: DataLoader, feature_dim: int, device: TorchDevice):
        self.num_classes = num_classes
        self.training_dataloader = training_dataloader

        self._construct_all_class_clusters(num_classes, training_dataloader)

    def _construct_all_class_clusters(self, num_classes: int, training_dataloader: DataLoader):
        # Initialize empty clusters for each class
        { i: GaussianCluster(feature_dim=self.feature_dim, class_id=i, device=self.device) for i in range(num_classes) }