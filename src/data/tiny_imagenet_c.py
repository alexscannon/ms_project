import os
from torchvision import datasets, transforms
from omegaconf import DictConfig

class TinyImagenetCDataset():
    def __init__(self, config: DictConfig):
        self.config = config
        self.base_dir = config.continual_learning.tinyimagenetC_location
        self.corruption_data_path = os.path.join(self.base_dir, config.continual_learning.corruption_type)

    def get_dataset(self, severity_level: int):
        severity_level_str = str(severity_level)
        return datasets.ImageFolder(
            root=os.path.join(self.corruption_data_path, severity_level_str),
            transform=self.transform()
        )

    def transform(self):
        return transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.continual_learning.tinyimagenetC_mean,
                std=self.config.continual_learning.tinyimagenetC_std
            )
        ])