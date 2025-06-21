# Cell 3: Import necessary libraries
import random
import torch
import os
import zipfile
import urllib.request
import numpy as np
from collections import defaultdict
import logging


# Cell 4: Configuration
TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # Or use .tar.gz if preferred

DRIVE_MOUNT_POINT = '/home/alex/data' # Optional: Google Drive mount point 
DATA_DIR = DRIVE_MOUNT_POINT + '/tiny-imagenet-200'
SAVE_DIR = DATA_DIR + '/preprocessed_tinyimagenet'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Data split ratios
IND_CLASS_RATIO = 0.80  # 80% of classes for In-Distribution
PRETRAIN_EXAMPLE_RATIO = 0.75  # 75% of examples from ID classes for actual pretraining

# Cell 5: Setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)
print(f"Using device: {DEVICE}")

# Cell 6: Download + extract Tiny Imagenet
# Create a directory for our dataset
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Data will be saved in: {DATA_DIR}")

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Preprocessed data will be saved in: {SAVE_DIR}")


# URL for the Tiny-ImageNet dataset
zip_path = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')

# Download the dataset if it doesn't exist
if not os.path.exists(zip_path):
    print("Downloading Tiny-ImageNet dataset...")

    # Create a progress bar for download
    def report_progress(block_num, block_size, total_size):
        progress = float(block_num * block_size) / float(total_size) * 100.0
        print(f"\rDownloading: {progress:.2f}%", end="")

    # Download with progress reporting
    urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path, reporthook=report_progress)
    print("\nDownload complete!")
else:
    print("Dataset already downloaded.")

# Extract the dataset if not already extracted
extract_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200')
if not os.path.exists(extract_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete!")
else:
    print("Dataset already extracted.")

# Basic validation to check the dataset structure
train_dir = os.path.join(extract_dir, 'train')
val_dir = os.path.join(extract_dir, 'test')

if os.path.exists(train_dir) and os.path.exists(val_dir):
    # Count the number of classes in training set
    train_classes = os.listdir(train_dir)
    print(f"Number of classes in training set: {len(train_classes)}")

    # Check a few example classes
    print(f"Example classes: {train_classes[:5]}")

    # Check the structure of one class
    example_class = train_classes[0]
    example_class_dir = os.path.join(train_dir, example_class)
    example_images_dir = os.path.join(example_class_dir, 'images')
    example_images = os.listdir(example_images_dir)

    print(f"Number of images in {example_class}: {len(example_images)}")
    print(f"Example image paths: {example_images[:3]}")
    print("Dataset structure validation complete!")
else:
    print("Dataset structure seems incorrect. Please check the extraction.")