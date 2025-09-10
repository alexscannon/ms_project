from torchvision import datasets, transforms
from omegaconf import DictConfig
import torch

class CIFAR100Dataset():
    def __init__(self, config: DictConfig):
        self.config = config

        self.train = datasets.CIFAR100(
            root=self.config.data.location,
            train=True,
            download=True,
            transform=self.transform()
        )

        self.val = datasets.CIFAR100(
            root=self.config.data.location,
            train=False,
            download=True,
            transform=self.transform()
        )

        self.all_data = torch.utils.data.ConcatDataset([self.train, self.val])

    def transform(self):
        if self.config.model.backbone.name == 'vit':
            transform = transforms.Compose([
                transforms.Resize(size=self.config.data.image_size), # Resize to the input size of the ViT model
                transforms.ToTensor(),
                transforms.Normalize(self.config.data.mean, self.config.data.std) # Normalize the data
            ])

        elif self.config.model.backbone.name == 'dinov2':
            # If the experiment is using the DINOv2 model, the images need to be transformed to match the expected input by DINOv2.
            transform = transforms.Compose([
                transforms.Resize(
                    size=self.config.model.backbone.expected_input_size,
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(size=self.config.model.backbone.center_crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.model.backbone.mean,
                    std=self.config.model.backbone.std
                ),
            ])

        else:
            raise ValueError(f"Model backbone {self.config.model.backbone.name} not supported")

        return transform