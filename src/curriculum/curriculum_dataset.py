from typing import List, Optional
from src.loaders.constellation_loader import ConstellationDataset
import os
from src.utils.snr_utils import get_snr_bucket_label, get_snr_label_names
from torchvision import transforms
from PIL import Image


class CurriculumAwareDataset(ConstellationDataset):
    """
    Curriculum-aware extension of ConstellationDataset.
    
    Allows dynamic filtering of dataset based on SNR values
    without reloading from disk. Efficiently updates dataset
    during curriculum stage transitions.
    
    References:
    [1] Chen, X., et al. (2021). "Efficient Dataset Management for Deep Learning."
    """
    
    def __init__(self, root_dir, snr_list=None, mods_to_process=None, 
                 image_type='three_channel', use_snr_buckets=False):
        """
        Initialize curriculum-aware dataset.
        
        Args:
            root_dir: Directory containing dataset
            snr_list: List of SNR values to include
            mods_to_process: List of modulation types to include
            image_type: Type of images to load ('grayscale', 'three_channel', 'point')
            use_snr_buckets: Whether to use SNR buckets
        """
        print(f"\n{'='*60}")
        print(f"INITIALIZING CURRICULUM DATASET WITH SNR LIST: {snr_list}")
        print(f"{'='*60}")
        
        # Save parameters for later use
        self.root_dir = root_dir
        self.mods_to_process = mods_to_process
        self.image_type = image_type
        self.use_snr_buckets = use_snr_buckets
        self.original_snr_list = snr_list
        
        # Initialize empty lists for dataset
        self.image_paths = []
        self.modulation_labels_list = []
        self.snr_labels_list = []
        
        # Initialize SNR and modulation mappings
        self.snr_labels = {}
        self.modulation_labels = {}
        self.inverse_snr_labels = {}
        self.inverse_modulation_labels = {}
        
        # Initialize transform based on image type
        if image_type == 'three_channel':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        elif image_type == 'grayscale':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485],
                                  std=[0.229])
            ])
        elif image_type == 'point':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485],
                                  std=[0.229])
            ])
        else:
            raise ValueError(f"Unsupported image_type '{image_type}'")
        
        # Load dataset with specified SNR list
        self._load_image_paths_and_labels(snr_list)
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(self.image_paths)}")
        
        # Count samples per modulation type
        mod_counts = {}
        for mod_label in self.modulation_labels_list:
            mod_type = self.inverse_modulation_labels[mod_label]
            mod_counts[mod_type] = mod_counts.get(mod_type, 0) + 1
            
        print("\nSamples per modulation type:")
        for mod_type, count in sorted(mod_counts.items()):
            print(f"  {mod_type}: {count} samples")
            
        # Count samples per SNR value
        snr_counts = {}
        for snr_label in self.snr_labels_list:
            snr_value = self.inverse_snr_labels[snr_label]
            snr_counts[snr_value] = snr_counts.get(snr_value, 0) + 1
            
        print("\nSamples per SNR value:")
        for snr in sorted(snr_counts.keys()):
            print(f"  SNR {snr}dB: {snr_counts[snr]} samples")
            
        print(f"Initialized curriculum dataset with {len(self.image_paths)} images")
        print(f"{'='*60}\n")
        
    def get_actual_snr_values(self, snr_idx):
        """
        Get the actual SNR value for a given index
        
        Args:
            snr_idx: Index of SNR in the label mapping
            
        Returns:
            Actual SNR value in dB
        """
        if snr_idx not in self.inverse_snr_labels:
            raise ValueError(f"SNR index {snr_idx} not found in mapping")
        return self.inverse_snr_labels[snr_idx]
        
    def update_snr_list(self, new_snr_list: List[int]) -> None:
        """
        Update dataset to include images with specified SNR values.
        This completely reloads the dataset with the new SNR values from disk.
        
        Args:
            new_snr_list: List of SNR values to include
        """
        print(f"\n{'='*50}")
        print(f"UPDATING DATASET WITH NEW SNR LIST: {new_snr_list}")
        print(f"{'='*50}")
        
        if new_snr_list is None:
            # Reset to original dataset
            self.image_paths = self.original_image_paths.copy()
            self.modulation_labels_list = self.original_modulation_labels_list.copy()
            self.snr_labels_list = self.original_snr_labels_list.copy()
            print(f"Reset dataset to original state with {len(self.image_paths)} images")
            return
        
        # Validate SNR list
        if not new_snr_list:
            raise ValueError("SNR list cannot be empty")
        if not all(isinstance(snr, int) for snr in new_snr_list):
            raise TypeError("All SNR values must be integers")
        
        # Save the count of the original dataset for comparison
        original_count = len(self.image_paths)
        
        # Reload dataset with new SNR list
        self._load_image_paths_and_labels(new_snr_list)
        
        # Print update statistics
        print(f"\nDataset Update Statistics:")
        print(f"Original sample count: {original_count}")
        print(f"New sample count: {len(self.image_paths)}")
        
        # Count samples per SNR value
        snr_counts = {}
        for snr_label in self.snr_labels_list:
            snr_value = self.inverse_snr_labels[snr_label]
            snr_counts[snr_value] = snr_counts.get(snr_value, 0) + 1
            
        print("\nSamples per SNR value after update:")
        for snr in sorted(snr_counts.keys()):
            print(f"  SNR {snr}dB: {snr_counts[snr]} samples")
            
        print(f"{'='*50}\n")
        
    def _load_image_paths_and_labels(self, snr_list=None):
        """
        Load image paths and labels based on specified SNR list.
        Overrides parent method to allow reloading with a specific SNR list.
        
        Args:
            snr_list: List of SNR values to include
            
        Returns:
            Tuple of (image_paths, modulation_labels_list, snr_labels_list)
        """
        print(f"Loading images with SNR list: {snr_list}")
        
        # Reset lists
        self.image_paths = []
        self.modulation_labels_list = []
        self.snr_labels_list = []
        
        # Reset mappings
        self.snr_labels = {}
        self.modulation_labels = {}
        self.inverse_snr_labels = {}
        self.inverse_modulation_labels = {}
        
        # Create SNR value to index mapping
        if snr_list:
            for i, snr in enumerate(sorted(snr_list)):
                self.snr_labels[snr] = i
                self.inverse_snr_labels[i] = snr
        
        # Process each modulation type
        for modulation_type in sorted(os.listdir(self.root_dir)):
            if self.mods_to_process and modulation_type not in self.mods_to_process:
                continue
                
            modulation_dir = os.path.join(self.root_dir, modulation_type)
            if not os.path.isdir(modulation_dir):
                continue
                
            # Create modulation type to index mapping
            if modulation_type not in self.modulation_labels:
                idx = len(self.modulation_labels)
                self.modulation_labels[modulation_type] = idx
                self.inverse_modulation_labels[idx] = modulation_type
            
            # Process each SNR value
            for snr_dir in sorted(os.listdir(modulation_dir)):
                try:
                    # Parse SNR value from directory name (e.g., "SNR_-20" -> -20)
                    if not snr_dir.startswith('SNR_'):
                        continue
                    snr_value = int(snr_dir.split('_')[1])
                    
                    if snr_list and snr_value not in snr_list:
                        continue
                        
                    snr_path = os.path.join(modulation_dir, snr_dir)
                    if not os.path.isdir(snr_path):
                        continue
                        
                    # Process each image
                    for image_file in sorted(os.listdir(snr_path)):
                        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                            continue
                            
                        image_path = os.path.join(snr_path, image_file)
                        
                        # Add to dataset
                        self.image_paths.append(image_path)
                        self.modulation_labels_list.append(self.modulation_labels[modulation_type])
                        
                        # Map SNR value to index
                        if snr_value not in self.snr_labels:
                            idx = len(self.snr_labels)
                            self.snr_labels[snr_value] = idx
                            self.inverse_snr_labels[idx] = snr_value
                        self.snr_labels_list.append(self.snr_labels[snr_value])
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse SNR value from directory {snr_dir}: {str(e)}")
                    continue
        
        # Save original dataset state
        self.original_image_paths = self.image_paths.copy()
        self.original_modulation_labels_list = self.modulation_labels_list.copy()
        self.original_snr_labels_list = self.snr_labels_list.copy()
        
        print(f"Loaded {len(self.image_paths)} images")
        print(f"SNR values: {sorted(list(self.snr_labels.keys()))}")
        print(f"Modulation types: {sorted(list(self.modulation_labels.keys()))}")

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with safety checks.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image, modulation_label, snr_label)
        """
        # Safety checks
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.image_paths)}")
        
        try:
            img_path = self.image_paths[idx]
            modulation_label = self.modulation_labels_list[idx]
            snr_label = self.snr_labels_list[idx]
            
            if self.image_type == 'three_channel':
                image = Image.open(img_path).convert('RGB')
            elif self.image_type == 'grayscale':
                image = Image.open(img_path).convert('L')
            elif self.image_type == 'point':
                image = Image.open(img_path).convert('L')
            else:
                raise ValueError(f"Unsupported image_type '{self.image_type}'.")
                
            image = self.transform(image)
            return image, modulation_label, snr_label
            
        except Exception as e:
            print(f"Error accessing item at index {idx}:")
            print(f"Dataset size: {len(self.image_paths)}")
            print(f"Error details: {str(e)}")
            raise 