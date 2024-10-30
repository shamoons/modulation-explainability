# src/constellation_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.snr_utils import get_snr_bucket_label, get_snr_label_names


class ConstellationDataset(Dataset):
    def __init__(self, root_dir, snr_list=None, mods_to_process=None, image_type='three_channel', use_snr_buckets=False):
        self.root_dir = root_dir
        self.mods_to_process = mods_to_process if mods_to_process is not None else []  # If not provided, load all modulations
        self.image_type = image_type
        self.use_snr_buckets = use_snr_buckets

        # Ensure snr_list is a list of integers
        if snr_list is not None:
            self.snr_list = sorted([int(snr) for snr in snr_list])
        else:
            self.snr_list = None  # Load all SNRs

        self.modulation_labels = {}
        self.inverse_modulation_labels = {}
        self.snr_labels = {}
        self.inverse_snr_labels = {}

        # Load image paths and labels
        self.image_paths, self.modulation_labels_list, self.snr_labels_list = self._load_image_paths_and_labels()

        # Default transform applied to all images
        if self.image_type == 'three_channel':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
            ])
        elif self.image_type == 'grayscale':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0], std=[1.0])
            ])
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'.")

    def _load_image_paths_and_labels(self):
        image_paths = []
        modulation_labels_list = []
        snr_labels_list = []
        snr_values_set = set()
        modulation_types_set = set()

        for modulation_type in sorted(os.listdir(self.root_dir)):  # Sorted for consistency
            modulation_dir = os.path.join(self.root_dir, modulation_type)
            if os.path.isdir(modulation_dir):
                if self.mods_to_process and modulation_type not in self.mods_to_process:
                    continue
                modulation_types_set.add(modulation_type)
                for snr_dir in sorted(os.listdir(modulation_dir)):
                    snr_path = os.path.join(modulation_dir, snr_dir)
                    if os.path.isdir(snr_path):
                        snr_value = int(snr_dir.split('_')[1])  # Extract SNR value
                        if self.snr_list is None or snr_value in self.snr_list:
                            snr_values_set.add(snr_value)
                            for img_name in os.listdir(snr_path):
                                if img_name.endswith('.png') and img_name.startswith(self.image_type):
                                    img_path = os.path.join(snr_path, img_name)
                                    image_paths.append(img_path)
                                    modulation_labels_list.append(modulation_type)

                                    # Optionally bucket SNR values (store the bucket label)
                                    if self.use_snr_buckets:
                                        snr_bucket_label = get_snr_bucket_label(snr_value)
                                        snr_labels_list.append(snr_bucket_label)
                                    else:
                                        snr_labels_list.append(snr_value)

        # Create label mappings for modulations
        modulation_types = sorted(list(modulation_types_set))
        self.modulation_labels = {mod: idx for idx, mod in enumerate(modulation_types)}
        self.inverse_modulation_labels = {idx: mod for mod, idx in self.modulation_labels.items()}

        # Create label mappings for SNRs
        if self.use_snr_buckets:
            # Get bucket names like ['low', 'medium', 'high']
            snr_label_names = get_snr_label_names()
            self.snr_labels = {label: idx for idx, label in enumerate(snr_label_names)}
            self.inverse_snr_labels = {idx: label for idx, label in enumerate(snr_label_names)}
        else:
            # Use raw SNR values
            snr_values = sorted(list(snr_values_set))
            self.snr_labels = {snr: idx for idx, snr in enumerate(snr_values)}
            self.inverse_snr_labels = {idx: snr for idx, snr in enumerate(snr_values)}

        # Convert modulation_labels_list and snr_labels_list from labels to indices
        modulation_labels_list = [self.modulation_labels[mod] for mod in modulation_labels_list]
        snr_labels_list = [self.snr_labels[snr] for snr in snr_labels_list]

        return image_paths, modulation_labels_list, snr_labels_list

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        modulation_label = self.modulation_labels_list[idx]
        snr_label = self.snr_labels_list[idx]

        if self.image_type == 'three_channel':
            image = Image.open(img_path).convert('RGB')
        elif self.image_type == 'grayscale':
            image = Image.open(img_path).convert('L')
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'.")

        image = self.transform(image)
        return image, modulation_label, snr_label
