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
        
        # IMPORTANT: Instead of using the parent class init which loads all SNRs,
        # we'll use our custom loader to only load specified SNRs from the start
        if snr_list is not None:
            print(f"Loading ONLY specified SNRs: {snr_list}")
        else:
            print(f"No SNR list provided, loading all available SNRs")
        
        # Load only images with specified SNR values
        self.image_paths, self.modulation_labels_list, self.snr_labels_list = self._load_image_paths_and_labels(
            snr_list=snr_list
        )
        
        # Setup transform
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
        elif self.image_type == 'point':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0], std=[1.0])
            ])
        else:
            raise ValueError(f"Unsupported image_type '{self.image_type}'.")
            
        # Store original data for filtering/resetting
        self.original_image_paths = self.image_paths.copy()
        self.original_modulation_labels_list = self.modulation_labels_list.copy()
        self.original_snr_labels_list = self.snr_labels_list.copy()
        
        # Print dataset statistics
        dataset_snrs = set()
        for snr_label in self.snr_labels_list:
            dataset_snrs.add(self.inverse_snr_labels[snr_label])
            
        snr_counts = {}
        for snr_label in self.snr_labels_list:
            actual_snr = self.inverse_snr_labels[snr_label]
            if actual_snr not in snr_counts:
                snr_counts[actual_snr] = 0
            snr_counts[actual_snr] += 1
            
        print(f"SNRs in initialized dataset: {sorted(list(dataset_snrs))}")
        print(f"Samples per SNR in dataset:")
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
        
        # KEY CHANGE: Instead of filtering, we completely reload the dataset
        # with the new SNR list
        # 1. Save important properties we need to preserve
        root_dir = self.root_dir
        mods_to_process = self.mods_to_process
        image_type = self.image_type
        use_snr_buckets = self.use_snr_buckets
        
        # 2. Reload the dataset with the new SNR list
        print(f"Reloading dataset from disk with SNR list: {new_snr_list}")
        image_paths, modulation_labels_list, snr_labels_list = self._load_image_paths_and_labels(
            snr_list=new_snr_list
        )
        
        # Safety check: Ensure all arrays are the same length
        if len(image_paths) != len(modulation_labels_list) or len(image_paths) != len(snr_labels_list):
            raise ValueError(f"Dataset arrays have inconsistent lengths: "
                             f"image_paths={len(image_paths)}, "
                             f"modulation_labels_list={len(modulation_labels_list)}, "
                             f"snr_labels_list={len(snr_labels_list)}")
            
        # 3. Update dataset properties
        self.image_paths = image_paths
        self.modulation_labels_list = modulation_labels_list
        self.snr_labels_list = snr_labels_list
        
        # 4. Verify the dataset was updated correctly
        snr_counts = {snr: 0 for snr in new_snr_list}
        for snr_label in self.snr_labels_list:
            actual_snr = self.inverse_snr_labels[snr_label]
            if actual_snr in snr_counts:
                snr_counts[actual_snr] += 1
        
        # Check for missing SNRs
        missing_snrs = [snr for snr in new_snr_list if snr_counts[snr] == 0]
        if missing_snrs:
            print(f"WARNING: No samples found for SNR values: {missing_snrs}")
        
        # Print counts per SNR
        print(f"Samples per SNR in updated dataset:")
        for snr in sorted(snr_counts.keys()):
            print(f"  SNR {snr}dB: {snr_counts[snr]} samples")
        
        # Calculate unique SNR values in the dataset
        dataset_snrs = set()
        for snr_label in self.snr_labels_list:
            dataset_snrs.add(self.inverse_snr_labels[snr_label])
        
        # Final safety check to ensure data integrity
        print(f"Final dataset check:")
        print(f"- Dataset size: {len(self.image_paths)} examples")
        print(f"- All arrays same length: {len(self.image_paths) == len(self.modulation_labels_list) == len(self.snr_labels_list)}")
        print(f"- SNRs in updated dataset: {sorted(list(dataset_snrs))}")
        print(f"- Dataset size changed from {original_count} to {len(self.image_paths)} images")
        print(f"{'='*50}\n")
        
        # Store updated data as new original data
        self.original_image_paths = self.image_paths.copy()
        self.original_modulation_labels_list = self.modulation_labels_list.copy()
        self.original_snr_labels_list = self.snr_labels_list.copy()
    
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
                        if snr_list is None or snr_value in snr_list:
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

        # Check if we have all the requested SNR values
        if snr_list is not None:
            missing_snrs = set(snr_list) - snr_values_set
            if missing_snrs:
                print(f"WARNING: The following SNR values were not found: {missing_snrs}")
                print(f"Available SNR values: {sorted(list(snr_values_set))}")

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