from torchvision import datasets, transforms
from omegaconf import DictConfig
import numpy as np

class CIFAR100Dataset():
    def __init__(self, config: DictConfig):
        self.config = config

        self.train = datasets.CIFAR100(
            root='./data/cifar100',
            train=True,
            download=True,
            transform=self.transform
        )

        self.test = datasets.CIFAR100(
            root='./data/cifar100',
            train=False,
            download=True,
            transform=self.transform
        )

        self.all_data = np.concatenate((self.train, self.test))

    def transform(self):
        return transforms.Compose([
            transforms.Resize(self.config.model.backbone.image_size), # Resize to the input size of the ViTmodel
            transforms.ToTensor(),
            transforms.Normalize(self.config.data.mean, self.config.data.std) # Normalize the data
        ])