import requests
import os
import tarfile
from tqdm import tqdm

def download_file(url: str, local_download_path: str) -> bool:
    """
    Downloads a file from a URL with a progress bar.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path to save the file.

    Returns:
        bool: True if the file exists or was successfully downloaded, False otherwise.
    """
    # Check if the file already exists
    if os.path.exists(local_download_path):
        print(f"File at '{local_download_path}' already exists. Skipping download.")
        return True

    print(f"Downloading '{local_download_path}' from '{url}'...")
    try:
        # Stream the download to handle large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes

            # Get the total file size from headers
            total_size = int(r.headers.get('content-length', 0))

            # Use tqdm for a progress bar
            with open(local_download_path, 'wb') as f, tqdm(
                desc=local_download_path,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        size = f.write(chunk)
                        bar.update(size)
        print(f"\nSuccessfully downloaded '{local_download_path}'.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        # Clean up partially downloaded file
        if os.path.exists(local_download_path):
            os.remove(local_download_path)
        return False

def unpack_tar_file(tar_path, extract_path='.'):
    """
    Unpacks a .tar file to a specified directory.

    Args:
        tar_path (str): The path to the .tar file.
        extract_path (str): The directory to extract the files to.

    Returns:
        bool: True if unpacking was successful, False otherwise.
    """
    if not tarfile.is_tarfile(tar_path):
        print(f"Error: '{tar_path}' is not a valid tar file.")
        return False

    print(f"Unpacking '{tar_path}'...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully unpacked to '{extract_path}'.")
        return True
    except tarfile.TarError as e:
        print(f"Error unpacking tar file: {e}")
        return False

if __name__ == "__main__":
    DATA_DIR = "/home/alex/data"  # Directory where you want to save the dataset

    # CIFAR-100-C dataset details
    # DATASET_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar" # URL of the CIFAR-100-C dataset on Zenodo
    # LOCAL_FILE_NAME = os.path.join(DATA_DIR, "CIFAR-100-C.tar") # The path with name you want to save the file as
    # EXTRACT_DIRECTORY = os.path.join(DATA_DIR, "CIFAR-100-C") # Directory to extract the contents to the same directory as the tar file

    # Tiny-IMagenet-C dataset details
    DATASET_URL = "https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar" # URL of the CIFAR-100-C dataset on Zenodo
    LOCAL_FILE_NAME = os.path.join(DATA_DIR, "Tiny-ImageNet-C.tar") # The path with name you want to save the file as
    EXTRACT_DIRECTORY = os.path.join(DATA_DIR, "Tiny-ImageNet-C") # Directory to extract the contents to the same directory as the tar file

    # Step 1: Download the file.
    if download_file(DATASET_URL, LOCAL_FILE_NAME):
        # Step 2: Unpack the downloaded file.
        if unpack_tar_file(LOCAL_FILE_NAME, EXTRACT_DIRECTORY):
            # Step 3: Clean up by removing the .tar file.
            try:
                os.remove(LOCAL_FILE_NAME)
                print(f"Removed archive '{LOCAL_FILE_NAME}'.")
            except OSError as e:
                print(f"Error removing archive file: {e}")