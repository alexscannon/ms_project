import torch
import time
from omegaconf import DictConfig

class DataStreamSimulator:
    """
    Simulates a data stream.
    Controls the streaming speed and batch size.
    """

    def __init__(self,
        dataset: torch.utils.data.Dataset,
        config: DictConfig = None
    ):

        self.dataset = dataset
        self.batch_size = config.clustering.batch_size
        self.stream_speed = config.clustering.stream_speed

        # Create data loader
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )

        self.total_samples = len(dataset)

    def stream(self):
        """
        Generator that yields batches of data with optional delay.
        Simulates a real-time data stream.
        """
        for batch_idx, (images, labels) in enumerate(self.dataloader):
            # Simulate streaming delay
            if self.stream_speed > 0:
                time.sleep(self.stream_speed)

            yield images, labels, batch_idx