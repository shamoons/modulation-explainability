# src/constellation_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset
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
            snr_list (list of int or None): List of SNR values to load. If None, load all SNRs.
            mods_to_process (list of str or None): List of modulation types to load. If None, load all modulations.
            image_type (str): Type of images to load ('three_channel' or 'grayscale').
        """
        self.root_dir = root_dir
        self.mods_to_process = mods_to_process if mods_to_process is not None else []  # If not provided, load all modulations
        self.image_type = image_type  # Image type to load

        # Ensure snr_list is a list of integers
        if snr_list is not None:
            self.snr_list = [int(snr) for snr in snr_list]
        else:
            self.snr_list = None  # Load all SNRs

        # Initialize empty dictionaries for label mappings
        self.modulation_labels = {}
        self.inverse_modulation_labels = {}
        self.snr_labels = {}
        self.inverse_snr_labels = {}

        # Load image paths and labels
        self.image_paths, self.modulation_labels_list, self.snr_labels_list = self._load_image_paths_and_labels()

        # Default transform applied to all images
        if self.image_type == 'three_channel':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to a standard size
                transforms.ToTensor(),  # Convert image to tensor (multi-channel)
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize 3-channel image between 0 and 1
            ])
        elif self.image_type == 'grayscale':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to a standard size
                transforms.ToTensor(),  # Convert image to tensor (single-channel)
                transforms.Normalize(mean=[0.0], std=[1.0])  # Normalize single-channel image between 0 and 1
            ])
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'. Supported types are 'three_channel' and 'grayscale'.")

    def _load_image_paths_and_labels(self):
        """
        Traverse the root directory and load image paths filtered by SNR, modulation type, and image type,
        also capturing the modulation type and SNR as labels.

        Returns:
            image_paths (list of str): List of paths to constellation images.
            modulation_labels_list (list of int): List of modulation labels.
            snr_labels_list (list of int): List of SNR labels.
        """
        image_paths = []
        modulation_labels_list = []
        snr_labels_list = []
        snr_values_set = set()
        modulation_types_set = set()

        # Traverse the directory structure
        for modulation_type in os.listdir(self.root_dir):
            modulation_dir = os.path.join(self.root_dir, modulation_type)
            if os.path.isdir(modulation_dir):  # Skip non-directory files
                # Check if the modulation type is in the specified list
                if self.mods_to_process and modulation_type not in self.mods_to_process:
                    continue
                modulation_types_set.add(modulation_type)
                for snr_dir in os.listdir(modulation_dir):
                    snr_path = os.path.join(modulation_dir, snr_dir)
                    if os.path.isdir(snr_path):
                        snr_value = int(snr_dir.split('_')[1])  # Extract SNR value
                        if self.snr_list is None or snr_value in self.snr_list:
                            snr_values_set.add(snr_value)
                            for img_name in os.listdir(snr_path):
                                if img_name.endswith('.png') and img_name.startswith(self.image_type):
                                    img_path = os.path.join(snr_path, img_name)
                                    image_paths.append(img_path)
                                    modulation_labels_list.append(modulation_type)  # Store modulation type as label
                                    snr_labels_list.append(snr_value)  # Store SNR value as label

        # Now create label mappings based on the collected modulation types and SNR values
        modulation_types = sorted(list(modulation_types_set))
        self.modulation_labels = {mod: idx for idx, mod in enumerate(modulation_types)}
        self.inverse_modulation_labels = {idx: mod for mod, idx in self.modulation_labels.items()}

        snr_values = sorted(list(snr_values_set))
        self.snr_labels = {snr: idx for idx, snr in enumerate(snr_values)}
        self.inverse_snr_labels = {idx: snr for snr, idx in self.snr_labels.items()}

        # Now convert modulation_labels_list and snr_labels_list from labels to indices
        modulation_labels_list = [self.modulation_labels[mod] for mod in modulation_labels_list]
        snr_labels_list = [self.snr_labels[snr] for snr in snr_labels_list]

        return image_paths, modulation_labels_list, snr_labels_list

    def __len__(self):
        """
        Return the total number of images loaded.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a specific image and labels, apply the default transformation (resizing, tensor conversion, normalization).

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: (Tensor image, int modulation_label, int snr_label)
        """
        img_path = self.image_paths[idx]
        modulation_label = self.modulation_labels_list[idx]
        snr_label = self.snr_labels_list[idx]

        # Load image
        if self.image_type == 'three_channel':
            image = Image.open(img_path).convert('RGB')  # Convert to RGB (3 channels)
        elif self.image_type == 'grayscale':
            image = Image.open(img_path).convert('L')  # Convert to grayscale (single channel)
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'.")

        # Apply the default transform (resize, to tensor, normalize)
        image = self.transform(image)

        return image, modulation_label, snr_label  # Return image, modulation label, and SNR label
