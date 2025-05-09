from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader creation."""
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")

class DatasetManager:
    """Manages dataset operations and curriculum updates."""
    
    def __init__(self, dataset: Any, config: DataLoaderConfig):
        """Initialize the dataset manager.
        
        Args:
            dataset: The dataset to manage
            config: Configuration for dataloader creation
        """
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
            
        self.dataset = dataset
        self.config = config
        
    def create_dataloader(self) -> DataLoader:
        """Create a DataLoader with the configured parameters."""
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.shuffle,
            pin_memory=self.config.pin_memory
        )
        
    def update_curriculum(self, min_snr: int, max_snr: int) -> None:
        """Update the dataset to include only samples within SNR range.
        
        Args:
            min_snr: Minimum SNR value to include
            max_snr: Maximum SNR value to include
        """
        if min_snr > max_snr:
            raise ValueError("Minimum SNR cannot be greater than maximum SNR")
        if min_snr < 0:
            raise ValueError("SNR values cannot be negative")
            
        self.dataset.update_curriculum(min_snr, max_snr)
        
    def get_snr_labels(self) -> Dict[int, int]:
        """Get the SNR labels mapping."""
        return self.dataset.snr_labels
        
    def dataset_length(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset) 