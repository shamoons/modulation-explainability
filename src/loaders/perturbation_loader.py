# src/loaders/perturbation_loader.py

import os
from loaders.constellation_loader import ConstellationDataset
from PIL import Image


class PerturbationDataset(ConstellationDataset):
    """
    Dataset class for loading pre-perturbed constellation images.
    """

    def __init__(
        self,
        root_dir,
        perturbation_dir,
        perturbation_type="top5_blackout",
        image_type="grayscale",
        snr_list=None,
        mods_to_process=None,
        use_snr_buckets=False,
    ):
        """
        Initialize the PerturbationDataset.

        Args:
            root_dir (str): Path to the original dataset directory (not used but kept for compatibility).
            perturbation_dir (str): Path to the directory containing perturbed images.
            perturbation_type (str): Type of perturbation ('top5_blackout' or 'bottom5_blackout').
            image_type (str): Type of image ('grayscale' or 'RGB').
            snr_list (list, optional): List of SNR values to include.
            mods_to_process (list, optional): List of modulation types to include.
            use_snr_buckets (bool): Whether to use SNR buckets.
        """
        super().__init__(root_dir, snr_list, mods_to_process, image_type, use_snr_buckets)

        self.root_dir = perturbation_dir  # Use the perturbed images directory
        self.perturbation_type = perturbation_type

        # Initialize the image paths and labels for perturbed images
        self.image_paths = []
        self.mod_labels = []
        self.snr_labels = []

        self._load_image_paths_and_labels()

    def _load_image_paths_and_labels(self):
        """
        Load image paths and corresponding labels from the perturbed images directory.
        """
        mods = self.mods_to_process if self.mods_to_process else os.listdir(self.root_dir)

        for mod in mods:
            mod_dir = os.path.join(self.root_dir, mod)
            if not os.path.isdir(mod_dir):
                continue  # Skip if not a directory

            for snr_dir in os.listdir(mod_dir):
                snr_value = int(snr_dir.split('_')[1])  # Extract SNR value from 'SNR_xx' directory
                if self.snr_list and snr_value not in self.snr_list:
                    continue  # Skip SNRs not in the list

                snr_dir_full = os.path.join(mod_dir, snr_dir)
                if not os.path.isdir(snr_dir_full):
                    continue  # Skip if not a directory

                for file_name in os.listdir(snr_dir_full):
                    if file_name.endswith(f"_{self.perturbation_type}.png"):
                        image_path = os.path.join(snr_dir_full, file_name)
                        self.image_paths.append(image_path)
                        self.mod_labels.append(self.modulation_labels[mod])
                        self.snr_labels.append(self.snr_labels[snr_value])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_tensor, modulation_label, snr_label)
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L' if self.image_type == 'grayscale' else 'RGB')

        # Apply transformation
        image = self.transform(image)

        modulation_label = self.mod_labels[idx]
        snr_label = self.snr_labels[idx]

        return image, modulation_label, snr_label
