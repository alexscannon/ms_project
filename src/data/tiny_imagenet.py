from torchvision import datasets, transforms
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

class TinyImagenetDataset():
    def __init__(self, config: DictConfig):
        # Currently DO NOT NEED SPECIAL TRANSFROMs because all this project is doing is testing.
        # training has already happened but might need future transforms for CL or Novelty Detection.
        self.config = config

        # Training Datasets
        self.train_dataset = datasets.ImageFolder(
            root=config.data.train_location,
            transform=self.transform()
        )
        self.train_ind_in_dataset = datasets.ImageFolder(
            root=config.data.train_ind_in_location,
            transform=self.transform()
        )
        self.train_ind_out_dataset = datasets.ImageFolder(
            root=config.data.train_ind_out_location,
            transform=self.transform()
        )
        self.train_ood_dataset = datasets.ImageFolder(
            root=config.data.train_ood_location,
            transform=self.transform()
        )

        # Validation Sets
        self.val_dataset = datasets.ImageFolder(
            root=config.data.val_location,
            transform=self.transform()
        )
        self.val_organized_dataset = datasets.ImageFolder(
            root=config.data.val_organized_location,
            transform=self.transform()
        )
        self.val_ind_dataset = datasets.ImageFolder(
            root=config.data.val_ind,
            transform=self.transform()
        )
        self.val_ood_dataset = datasets.ImageFolder(
            root=config.data.val_ood,
            transform=self.transform()
        )

        # ------ Complete OOD dateset
        self.ood_dataset = ConcatDataset([self.train_ood_dataset, self.val_ood_dataset])

        # ------ Test ------ #
        self.test_dataset = datasets.ImageFolder(
            root=config.data.test_location,
            transform=self.transform()
        )

    def transform(self):
        return transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.mean,
                std=self.config.data.std
            )
        ])