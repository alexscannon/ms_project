import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.inputs[idx], self.targets[idx]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        return (img, target)


    def __len__(self):
        return len(self.inputs)