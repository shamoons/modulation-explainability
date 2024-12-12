# src/loaders/perturbation_loader.py
import os
from torchvision import transforms
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
        use_snr_buckets=True,
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
        self.root_dir = perturbation_dir  # Use the perturbed images directory
        self.image_type = image_type
        self.snr_list = snr_list
        self.mods_to_process = mods_to_process
        self.use_snr_buckets = use_snr_buckets
        self.perturbation_type = perturbation_type

        # Initialize labels and mappings
        self.modulation_labels = {}
        self.inverse_modulation_labels = {}
        self.snr_labels = {}
        self.inverse_snr_labels = {}

        # Load image paths and labels
        self.image_paths, self.mod_labels, self.snr_labels = self._load_image_paths_and_labels()

        # Define the transformation
        if self.image_type == "grayscale":
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to [C, H, W] with values in [0, 1]
            ])
        elif self.image_type == "RGB":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unsupported image type: {self.image_type}")

    def _load_image_paths_and_labels(self):
        """
        Load image paths and corresponding labels from the perturbed images directory.
        """
        image_paths = []
        mod_labels_list = []
        snr_labels_list = []
        modulation_types_set = set()
        snr_values_set = set()

        mods = self.mods_to_process if self.mods_to_process else os.listdir(self.root_dir)

        for mod in sorted(mods):  # Sort for consistency
            mod_dir = os.path.join(self.root_dir, mod)
            if not os.path.isdir(mod_dir):
                continue  # Skip if not a directory

            modulation_types_set.add(mod)

            snr_dirs = os.listdir(mod_dir)
            for snr_dir in sorted(snr_dirs):
                try:
                    # Extract SNR value from directory name (e.g., 'SNR_-10' -> -10)
                    snr_value = int(snr_dir.split('_')[1])
                    if self.snr_list and snr_value not in self.snr_list:
                        continue  # Skip if not in the specified SNR list
                    snr_values_set.add(snr_value)

                    # Debug log
                    # print(f"Processing SNR value: {snr_value} / {mod}")
                except (IndexError, ValueError):
                    # print(f"Invalid SNR directory name: {snr_dir} / {mod}")  # Debugging invalid directory names
                    continue  # Skip invalid SNR directories

                if self.snr_list and snr_value not in self.snr_list:
                    continue  # Skip if not in the specified SNR list

                snr_values_set.add(snr_value)

                snr_dir_full = os.path.join(mod_dir, snr_dir)
                if not os.path.isdir(snr_dir_full):
                    continue  # Skip if not a directory

                # List all images with the specified perturbation type
                for file_name in os.listdir(snr_dir_full):
                    if file_name.endswith(f"_{self.perturbation_type}.png"):
                        image_path = os.path.join(snr_dir_full, file_name)
                        image_paths.append(image_path)
                        mod_labels_list.append(mod)
                        if self.use_snr_buckets:
                            from utils.snr_utils import get_snr_bucket_label
                            snr_bucket_label = get_snr_bucket_label(snr_value)
                            snr_labels_list.append(snr_bucket_label)
                        else:
                            snr_labels_list.append(snr_value)

        # Create modulation and SNR mappings
        modulation_types = sorted(list(modulation_types_set))
        self.modulation_labels = {mod: idx for idx, mod in enumerate(modulation_types)}
        self.inverse_modulation_labels = {idx: mod for mod, idx in self.modulation_labels.items()}

        if self.use_snr_buckets:
            from utils.snr_utils import get_snr_label_names
            snr_label_names = get_snr_label_names()  # E.g., ['low', 'medium', 'high']
            self.snr_labels = {label: idx for idx, label in enumerate(snr_label_names)}
            self.inverse_snr_labels = {idx: label for idx, label in enumerate(snr_label_names)}
        else:
            snr_values = sorted(list(snr_values_set))
            self.snr_labels = {snr: idx for idx, snr in enumerate(snr_values)}
            self.inverse_snr_labels = {idx: snr for idx, snr in enumerate(snr_values)}

        # Convert mod_labels_list and snr_labels_list to indices
        mod_labels_list = [self.modulation_labels[mod] for mod in mod_labels_list]
        try:
            snr_labels_list = [self.snr_labels[snr] for snr in snr_labels_list]
        except KeyError as e:
            raise KeyError(f"Unexpected SNR value: {e} not found in SNR labels mapping.\nAvailable SNR labels: {self.snr_labels.keys()}")  # Debugging statement

        return image_paths, mod_labels_list, snr_labels_list

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
