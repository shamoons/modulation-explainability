import pytest
import torch
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants for test fixtures
TEST_BATCH_SIZE = 8
TEST_INPUT_CHANNELS = 1
TEST_INPUT_SIZE = 32
TEST_NUM_CLASSES = 4  # Number of modulation classes
TEST_SNR_CLASSES = 5  # Number of SNR classes

@pytest.fixture
def mock_device():
    """Return device to use for testing."""
    return torch.device('cpu')


@pytest.fixture
def mock_model():
    """Create a mock model that returns dummy predictions."""
    model = MagicMock()
    
    # Configure the mock model to return dummy outputs when called
    def side_effect(x):
        batch_size = x.shape[0]
        # Generate dummy outputs: (modulation classification, SNR classification)
        mod_outputs = torch.randn(batch_size, TEST_NUM_CLASSES)  # 4 modulation classes
        snr_outputs = torch.randn(batch_size, TEST_SNR_CLASSES)  # 5 SNR classes/buckets
        return mod_outputs, snr_outputs
    
    model.side_effect = side_effect
    model.return_value = (torch.randn(TEST_BATCH_SIZE, TEST_NUM_CLASSES), 
                          torch.randn(TEST_BATCH_SIZE, TEST_SNR_CLASSES))
    
    # Add training/eval mode tracking
    model.training = True
    
    def train(mode=True):
        model.training = mode
        return model
    
    def eval():
        model.training = False
        return model
    
    model.train = train
    model.eval = eval
    
    return model


@pytest.fixture
def mock_criterion():
    """Create a mock loss criterion."""
    criterion = MagicMock()
    criterion.return_value = torch.tensor(0.5, requires_grad=True)
    return criterion


@pytest.fixture
def mock_criterion_dynamic():
    """Create a mock dynamic weighting criterion."""
    criterion = MagicMock()
    
    # Return a total loss and weights
    def side_effect(losses):
        weights = torch.ones(len(losses)) / len(losses)
        total_loss = sum(w * l for w, l in zip(weights, losses))
        return total_loss, weights
        
    criterion.side_effect = side_effect
    criterion.return_value = (torch.tensor(0.5, requires_grad=True), torch.tensor([0.5, 0.5]))
    
    return criterion


@pytest.fixture
def test_input_batch():
    """Create a test input batch for model testing."""
    # Create a batch of 8 samples, 1 channel, 32x32 size (typical constellation diagram size)
    return torch.randn(TEST_BATCH_SIZE, TEST_INPUT_CHANNELS, TEST_INPUT_SIZE, TEST_INPUT_SIZE)


@pytest.fixture
def test_labels():
    """Generate test modulation and SNR labels"""
    mod_labels = torch.randint(0, TEST_NUM_CLASSES, (TEST_BATCH_SIZE,))
    snr_labels = torch.randint(0, TEST_SNR_CLASSES, (TEST_BATCH_SIZE,))
    return mod_labels, snr_labels


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    dataloader = MagicMock()
    
    # Create sample batches
    batches = [
        (torch.randn(TEST_BATCH_SIZE, TEST_INPUT_CHANNELS, TEST_INPUT_SIZE, TEST_INPUT_SIZE), 
         torch.randint(0, TEST_NUM_CLASSES, (TEST_BATCH_SIZE,)),  # Modulation labels
         torch.randint(0, TEST_SNR_CLASSES, (TEST_BATCH_SIZE,)))  # SNR labels
        for _ in range(3)  # 3 batches
    ]
    
    # Configure the dataloader to iterate through batches
    dataloader.__iter__.return_value = iter(batches)
    dataloader.__len__.return_value = len(batches)
    
    return dataloader


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = MagicMock()
    
    def zero_grad():
        pass
    
    def step():
        pass
    
    optimizer.zero_grad = MagicMock(side_effect=zero_grad)
    optimizer.step = MagicMock(side_effect=step)
    
    return optimizer


@pytest.fixture
def mock_scheduler():
    """Create a mock learning rate scheduler."""
    scheduler = MagicMock()
    
    def step():
        pass
    
    scheduler.step = MagicMock(side_effect=step)
    
    return scheduler


@pytest.fixture
def mock_curriculum_manager():
    """Create a mock curriculum learning manager."""
    manager = MagicMock()
    
    # Current curriculum state
    manager.current_stage = 1
    manager.current_snr_list = [-6, -4, -2, 0, 2]
    manager.max_stage = 3
    
    # Progress to next stage if accuracy meets threshold
    def should_progress(snr_accuracy):
        return snr_accuracy >= 75.0
    
    # Get next curriculum stage
    def get_next_stage():
        if manager.current_stage < manager.max_stage:
            return manager.current_stage + 1, [-8, -6, -4, -2, 0, 2, 4]
        return manager.current_stage, manager.current_snr_list
    
    manager.should_progress = MagicMock(side_effect=should_progress)
    manager.get_next_stage = MagicMock(side_effect=get_next_stage)
    
    return manager


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = MagicMock()
    
    # Dataset metadata
    dataset.mod_classes = ['BPSK', 'QPSK', '8PSK', 'QAM16']
    dataset.snr_values = [-6, -4, -2, 0, 2]
    dataset.snr_indices = [0, 1, 2, 3, 4]
    
    # Configure dataset length and getitem
    dataset.__len__.return_value = 100
    
    def getitem(idx):
        # Return (sample, mod_label, snr_label)
        return (
            torch.randn(TEST_INPUT_CHANNELS, TEST_INPUT_SIZE, TEST_INPUT_SIZE),
            torch.randint(0, TEST_NUM_CLASSES, (1,)).item(),
            torch.randint(0, TEST_SNR_CLASSES, (1,)).item()
        )
    
    dataset.__getitem__.side_effect = getitem
    
    return dataset


@pytest.fixture
def mock_snr_mapping():
    """Create a mock SNR mapping for testing."""
    # Maps SNR indices to actual SNR values
    return {
        0: -6,
        1: -4,
        2: -2,
        3: 0,
        4: 2,
        5: 4,
        6: 6
    }


@pytest.fixture
def test_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir) 