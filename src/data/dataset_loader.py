from torchvision import datasets, transforms
import os

DATASET_REGISTRY = {
    'cifar100': datasets.CIFAR100,
    'tiny_imagenet': datasets.ImageFolder,
}

def create_transform(config):
    """Create a transform pipeline based on config."""
    transform_list = []

    # Add resize if image_size is specified
    if hasattr(config.dataset, 'image_size'):
        transform_list.append(
            transforms.Resize((config.dataset.image_size, config.dataset.image_size))
        )
    # Add normalization if mean and std are specified
    if hasattr(config.dataset, 'mean') and hasattr(config.dataset, 'std'):
        transform_list.append(
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std)
        )

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

def load_dataset(config):
    """Load dataset based on configuration and wrap in ContinualDataset."""
    dataset_name = config.dataset.name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not supported. "
                       f"Available datasets: {list(DATASET_REGISTRY.keys())}")

    # Create transform
    transform = create_transform(config)

    # Get dataset class
    dataset_class = DATASET_REGISTRY[dataset_name]

    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == 'imagenet':
        # ImageNet requires separate train and val directories
        train_dir = os.path.join(config.dataset.root, 'train')
        val_dir = os.path.join(config.dataset.root, 'val')

        if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
            raise ValueError(
                f"ImageNet directories not found at {config.dataset.root}. "
                "Expected structure: \n"
                "root/\n"
                "  train/\n"
                "    n01440764/\n"
                "      *.JPEG\n"
                "  val/\n"
                "    n01440764/\n"
                "      *.JPEG"
            )

        train_dataset = dataset_class(
            root=config.dataset.root,
            split='train',
            transform=transform
        )

        test_dataset = dataset_class(
            root=config.dataset.root,
            split='val',
            transform=transform
        )
    else:
        # Load train and test datasets using the root path from config
        train_dataset = dataset_class(
            root=config.dataset.root,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = dataset_class(
            root=config.dataset.root,
            train=False,
            download=True,
            transform=transform
        )

    # Wrap in ContinualDataset
    # Could move back to main.py if we want to separate the continual dataset wrapper from the dataset loading.
    continual_dataset = ContinualDataset(
        train_dataset,
        test_dataset,
        num_classes=config.dataset.num_classes
    )

    return continual_dataset