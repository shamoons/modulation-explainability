# src/constellation_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ConstellationDataset(Dataset):
    """
    PyTorch Dataset class to load constellation images from directories,
    optionally filtering by SNR. Images are loaded as grayscale (single channel).
    """

    def __init__(self, root_dir, snr_list=None, transform=None):
        """
        Args:
            root_dir (str): Root directory where the constellation images are stored.
            snr_list (list of str or None): List of SNR values to load. If None, load all SNRs.
            transform (callable, optional): Optional transform to be applied on a sample (image).
        """
        self.root_dir = root_dir
        self.snr_list = snr_list if snr_list is not None else []  # If not provided, load all SNRs
        self.transform = transform

        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        """
        Traverse the root directory and load image paths filtered by SNR (if applicable).

        Returns:
            image_paths (list of str): List of paths to constellation images.
        """
        image_paths = []

        # Traverse the directory structure
        for modulation_type in os.listdir(self.root_dir):
            modulation_dir = os.path.join(self.root_dir, modulation_type)
            if os.path.isdir(modulation_dir):
                for snr_dir in os.listdir(modulation_dir):
                    snr_value = snr_dir.split('_')[1]  # Extract SNR from directory name (e.g., "SNR_2" -> 2)
                    if not self.snr_list or snr_value in self.snr_list:
                        snr_path = os.path.join(modulation_dir, snr_dir)
                        for img_name in os.listdir(snr_path):
                            if img_name.endswith('.png'):  # Only load PNG images
                                img_path = os.path.join(snr_path, img_name)
                                image_paths.append(img_path)

        return image_paths

    def __len__(self):
        """
        Return the total number of images loaded.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a specific image and apply any specified transformations.

        Args:
            idx (int): Index of the image to load.

        Returns:
            sample (Tensor): The transformed image (grayscale, single-channel).
        """
        img_path = self.image_paths[idx]
        # Load image as grayscale (single channel)
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image


def get_constellation_dataloader(root_dir, snr_list=None, batch_size=64, shuffle=True, transform=None):
    """
    Function to create a DataLoader for the constellation dataset.

    Args:
        root_dir (str): Root directory where constellation images are stored.
        snr_list (list of str or None): List of SNR values to load. If None, load all SNRs.
        batch_size (int): Number of images per batch.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transformation for the images.

    Returns:
        DataLoader: PyTorch DataLoader for the constellation dataset.
    """
    dataset = ConstellationDataset(root_dir, snr_list=snr_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    # Example usage: Load all SNRs
    root_dir = "constellation"  # Replace with the actual directory where constellation images are stored
    snr_list = ['-10', '2']  # Example: Load only images with SNRs -10 and 2 (can be omitted to load all)

    # Example transformation (convert image to tensor and normalize)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a standard size
        transforms.ToTensor(),  # Convert image to tensor (single-channel)
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image between -1 and 1
    ])

    # Get DataLoader
    dataloader = get_constellation_dataloader(root_dir, snr_list=snr_list, batch_size=32, transform=transform)

    # Iterate through the DataLoader (for demonstration purposes)
    for images in dataloader:
        print(f"Batch of images: {images.size()}")  # Should print (batch_size, 1, 64, 64) for single-channel images
