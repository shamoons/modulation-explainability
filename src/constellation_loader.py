# src/constellation_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ConstellationDataset(Dataset):
    """
    PyTorch Dataset class to load constellation images from directories,
    optionally filtering by SNR, modulation type, and image type.
    Images are loaded as 3-channel (RGB) or single-channel (grayscale) based on image_type.
    """

    def __init__(self, root_dir, snr_list=None, mods_to_process=None, image_type='three_channel'):
        """
        Args:
            root_dir (str): Root directory where the constellation images are stored.
            snr_list (list of str or None): List of SNR values to load. If None, load all SNRs.
            mods_to_process (list of str or None): List of modulation types to load. If None, load all modulations.
            image_type (str): Type of images to load ('three_channel' or 'grayscale').
        """
        self.root_dir = root_dir
        self.snr_list = snr_list if snr_list is not None else []  # If not provided, load all SNRs
        self.mods_to_process = mods_to_process if mods_to_process is not None else []  # If not provided, load all modulations
        self.image_type = image_type  # Image type to load

        # Create a dictionary only for directories and save it as an attribute
        self.modulation_labels = {mod: idx for idx, mod in enumerate(os.listdir(self.root_dir)) if os.path.isdir(os.path.join(self.root_dir, mod))}

        # Print available modulation schemes
        print(f"Available modulation schemes: {list(self.modulation_labels.keys())}")

        # Internal method to fetch image paths and labels
        self.image_paths, self.labels = self._load_image_paths_and_labels()

        # Default transform applied to all images (Resizing, ToTensor, and Normalizing)
        if self.image_type == 'three_channel':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to a standard size
                transforms.ToTensor(),  # Convert image to tensor (multi-channel)
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize 3-channel image between -1 and 1
            ])
        elif self.image_type == 'grayscale':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to a standard size
                transforms.ToTensor(),  # Convert image to tensor (single-channel)
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize single-channel image between -1 and 1
            ])
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'. Supported types are 'three_channel' and 'grayscale'.")

    def _load_image_paths_and_labels(self):
        """
        Traverse the root directory and load image paths filtered by SNR, modulation type, and image type,
        also capturing the modulation type as a label.

        Returns:
            image_paths (list of str): List of paths to constellation images.
            labels (list of int): List of labels corresponding to modulation types.
        """
        image_paths = []
        labels = []

        # Traverse the directory structure
        for modulation_type in os.listdir(self.root_dir):
            modulation_dir = os.path.join(self.root_dir, modulation_type)
            if os.path.isdir(modulation_dir):  # Skip non-directory files like .DS_Store
                # Check if the modulation type is in the specified list
                if self.mods_to_process and modulation_type not in self.mods_to_process:
                    continue
                for snr_dir in os.listdir(modulation_dir):
                    snr_path = os.path.join(modulation_dir, snr_dir)
                    if os.path.isdir(snr_path):  # Ensure this is a directory
                        snr_value = snr_dir.split('_')[1]  # Extract SNR from directory name (e.g., "SNR_2" -> "2")
                        if not self.snr_list or snr_value in self.snr_list:
                            for img_name in os.listdir(snr_path):
                                if img_name.endswith('.png') and img_name.startswith(self.image_type):  # Filter by image_type
                                    img_path = os.path.join(snr_path, img_name)
                                    image_paths.append(img_path)
                                    labels.append(self.modulation_labels[modulation_type])  # Assign modulation label

        return image_paths, labels

    def __len__(self):
        """
        Return the total number of images loaded.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a specific image and label, apply the default transformation (resizing, tensor conversion, normalization).

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: (Tensor image, int label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        if self.image_type == 'three_channel':
            image = Image.open(img_path).convert('RGB')  # Convert to RGB (3 channels)
        elif self.image_type == 'grayscale':
            image = Image.open(img_path).convert('L')  # Convert to grayscale (single channel)
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'.")

        # Apply the default transform (resize, to tensor, normalize)
        image = self.transform(image)

        return image, label  # Return both image and label


def get_constellation_dataloader(root_dir, snr_list=None, mods_to_process=None, image_type='three_channel', batch_size=64, shuffle=True):
    """
    Function to create a DataLoader for the constellation dataset.

    Args:
        root_dir (str): Root directory where constellation images are stored.
        snr_list (list of str or None): List of SNR values to load. If None, load all SNRs.
        mods_to_process (list of str or None): List of modulation types to load. If None, load all modulations.
        image_type (str): Type of images to load ('three_channel' or 'grayscale').
        batch_size (int): Number of images per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader for the constellation dataset.
    """
    dataset = ConstellationDataset(root_dir, snr_list=snr_list, mods_to_process=mods_to_process, image_type=image_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    # Example usage: Load images with specified SNRs, modulations, and image type
    root_dir = "constellation"  # Replace with the actual directory where constellation images are stored
    snr_list = ['20', '30']  # Example: Load only images with SNRs 20 and 30 (can be omitted to load all)
    mods_to_process = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', 'QAM16', 'QAM64', 'QAM256']
    image_type = 'grayscale'  # Specify the image type to load ('three_channel' or 'grayscale')

    # Get DataLoader
    dataloader = get_constellation_dataloader(root_dir, snr_list=snr_list, mods_to_process=mods_to_process, image_type=image_type, batch_size=32)

    # Iterate through the DataLoader (for demonstration purposes)
    for images, labels in dataloader:
        print(f"Batch of images: {images.size()}, Batch of labels: {labels.size()}")  # Print image and label batch sizes
