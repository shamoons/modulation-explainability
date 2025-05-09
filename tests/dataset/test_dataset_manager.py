import pytest
import torch
from unittest.mock import MagicMock, patch
from src.dataset.dataset_manager import DatasetManager, DataLoaderConfig

@pytest.fixture
def mock_dataset():
    """Create a mock dataset with basic functionality."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.snr_labels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    # For DataLoader iteration
    dataset.__getitem__.side_effect = lambda idx: torch.tensor([idx])
    return dataset

@pytest.fixture
def dataloader_config():
    """Create a basic dataloader configuration."""
    return DataLoaderConfig(
        batch_size=32,
        num_workers=0,
        shuffle=True,
        pin_memory=True
    )

class TestDatasetManager:
    """Test suite for DatasetManager class."""
    
    def test_initialization(self, mock_dataset, dataloader_config):
        """Test that DatasetManager initializes correctly."""
        manager = DatasetManager(mock_dataset, dataloader_config)
        assert manager.dataset == mock_dataset
        assert manager.config == dataloader_config
        
    def test_create_dataloader(self, mock_dataset, dataloader_config):
        """Test that create_dataloader returns a DataLoader that yields correct batch size."""
        manager = DatasetManager(mock_dataset, dataloader_config)
        dataloader = manager.create_dataloader()
        batch = next(iter(dataloader))
        # batch is a tensor of shape [batch_size, 1] due to our __getitem__
        assert batch.shape[0] == dataloader_config.batch_size
        
    def test_update_curriculum(self, mock_dataset, dataloader_config):
        """Test that update_curriculum correctly filters dataset."""
        manager = DatasetManager(mock_dataset, dataloader_config)
        
        # Test updating to a specific SNR range
        manager.update_curriculum(min_snr=1, max_snr=3)
        mock_dataset.update_curriculum.assert_called_once_with(1, 3)
        
    def test_get_snr_labels(self, mock_dataset, dataloader_config):
        """Test that get_snr_labels returns correct labels."""
        manager = DatasetManager(mock_dataset, dataloader_config)
        labels = manager.get_snr_labels()
        assert labels == mock_dataset.snr_labels
        
    def test_dataset_length(self, mock_dataset, dataloader_config):
        """Test that dataset_length returns correct value."""
        manager = DatasetManager(mock_dataset, dataloader_config)
        length = manager.dataset_length()
        assert length == 100
        
    def test_invalid_config(self, mock_dataset):
        """Test that invalid config raises appropriate error."""
        with pytest.raises(ValueError):
            DataLoaderConfig(batch_size=0, num_workers=0, shuffle=True, pin_memory=True)  # Invalid batch size
            
    def test_empty_dataset(self, dataloader_config):
        """Test that empty dataset raises appropriate error."""
        empty_dataset = MagicMock()
        empty_dataset.__len__.return_value = 0
        
        with pytest.raises(ValueError):
            DatasetManager(empty_dataset, dataloader_config)
            
    def test_curriculum_update_validation(self, mock_dataset, dataloader_config):
        """Test that invalid SNR ranges raise appropriate error."""
        manager = DatasetManager(mock_dataset, dataloader_config)
        
        with pytest.raises(ValueError):
            manager.update_curriculum(min_snr=5, max_snr=3)  # min > max
            
        with pytest.raises(ValueError):
            manager.update_curriculum(min_snr=-1, max_snr=3)  # negative SNR 