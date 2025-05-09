import pytest
import torch
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import function to test
from src.training_constellation import train

from tests.utils.test_helpers import (
    capture_model_outputs,
    save_model_snapshot,
    compare_model_snapshots
)


@pytest.fixture
def snapshot_dir(tmp_path):
    """Create a temporary directory for snapshots."""
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    return str(snapshot_dir)


@pytest.mark.snapshot
def test_create_initial_model_snapshot(mock_model, test_input_batch, snapshot_dir):
    """
    Create an initial snapshot of model behavior for later comparison.
    
    This test should be run before refactoring to create baseline snapshots.
    """
    # Get model outputs
    outputs = capture_model_outputs(mock_model, test_input_batch)
    
    # Save snapshot
    snapshot_path = os.path.join(snapshot_dir, "model_initial_snapshot.json")
    save_model_snapshot(outputs, snapshot_path)
    
    # Verify snapshot file was created
    assert os.path.exists(snapshot_path), f"Snapshot file not created at {snapshot_path}"
    
    # This test just creates a baseline, actual comparisons will be done in other tests
    print(f"Created model snapshot at {snapshot_path}")


@pytest.mark.snapshot
def test_model_snapshot_consistency(mock_model, test_input_batch, snapshot_dir):
    """
    Test that model behavior remains consistent when producing the same outputs twice.
    
    This validates our snapshot comparison mechanism works properly.
    """
    # Create two snapshots with the same model and inputs
    # They should be identical or at least very similar
    snapshot1_path = os.path.join(snapshot_dir, "consistency_1.json")
    snapshot2_path = os.path.join(snapshot_dir, "consistency_2.json")
    
    # Force deterministic behavior
    torch.manual_seed(42)
    outputs1 = capture_model_outputs(mock_model, test_input_batch)
    save_model_snapshot(outputs1, snapshot1_path)
    
    torch.manual_seed(42)
    outputs2 = capture_model_outputs(mock_model, test_input_batch)
    save_model_snapshot(outputs2, snapshot2_path)
    
    # Compare snapshots
    identical, differences = compare_model_snapshots(snapshot1_path, snapshot2_path)
    
    # They should be identical
    assert identical, f"Model outputs should be identical for the same inputs. Differences: {differences}"


@pytest.fixture
def mock_training_batch():
    """Create a mock training batch for snapshot testing."""
    # Create batch of 8 samples
    inputs = torch.randn(8, 1, 32, 32)
    mod_labels = torch.randint(0, 4, (8,))
    snr_labels = torch.randint(0, 5, (8,))
    return inputs, mod_labels, snr_labels


@pytest.mark.snapshot
def test_training_batch_processing(mock_model, mock_criterion, mock_criterion_dynamic, mock_training_batch, snapshot_dir):
    """Test processing of a single training batch and capture outputs."""
    # Unpack mock batch
    inputs, mod_labels, snr_labels = mock_training_batch
    
    # Forward pass
    mod_outputs, snr_outputs = mock_model(inputs)
    
    # Calculate losses
    mod_loss = mock_criterion(mod_outputs, mod_labels)
    snr_loss = mock_criterion(snr_outputs, snr_labels)
    
    # Dynamic loss weighting
    total_loss, weights = mock_criterion_dynamic([mod_loss, snr_loss])
    
    # Capture outputs
    batch_outputs = {
        'mod_outputs': mod_outputs,
        'snr_outputs': snr_outputs,
        'mod_loss': mod_loss,
        'snr_loss': snr_loss,
        'total_loss': total_loss,
        'weights': weights
    }
    
    # Save snapshot
    snapshot_path = os.path.join(snapshot_dir, "training_batch_snapshot.json")
    save_model_snapshot(batch_outputs, snapshot_path)
    
    # Verify snapshot was created
    assert os.path.exists(snapshot_path), f"Snapshot file not created at {snapshot_path}"
    print(f"Created training batch snapshot at {snapshot_path}")


@pytest.mark.snapshot
def test_curriculum_advancement_snapshot(mock_curriculum_manager, snapshot_dir):
    """Test curriculum advancement behavior and snapshot the stage transition."""
    # Save current stage
    initial_stage = mock_curriculum_manager.current_stage
    initial_snr_list = mock_curriculum_manager.current_snr_list
    
    # Prepare curriculum snapshot
    curriculum_state = {
        'initial_stage': initial_stage,
        'initial_snr_list': initial_snr_list,
        'should_progress': mock_curriculum_manager.should_progress(80.0),  # 80% SNR accuracy
        'next_stage': mock_curriculum_manager.get_next_stage()
    }
    
    # Save snapshot
    snapshot_path = os.path.join(snapshot_dir, "curriculum_advancement_snapshot.json")
    save_model_snapshot(curriculum_state, snapshot_path)
    
    # Verify snapshot was created
    assert os.path.exists(snapshot_path), f"Snapshot file not created at {snapshot_path}"
    print(f"Created curriculum advancement snapshot at {snapshot_path}")


@pytest.mark.snapshot
@patch('src.training_constellation.train')
@patch('src.training_constellation.wandb.init')
@patch('src.training_constellation.wandb.log')
@patch('builtins.print')
def test_complete_training_snapshot(mock_print, mock_wandb_log, mock_wandb_init, mock_train, mock_model, mock_criterion, mock_criterion_dynamic, mock_dataloader, snapshot_dir):
    """Create a snapshot of complete training behavior."""
    # Mock the train function to return a predefined output
    mock_train.return_value = "checkpoints/best_model.pth"
    
    # Mock wandb.init to return a MagicMock
    mock_wandb = MagicMock()
    mock_wandb_init.return_value = mock_wandb
    
    # Mock wandb.log to do nothing
    mock_wandb_log.return_value = None
    
    # Mock print to do nothing
    mock_print.return_value = None
    
    # Mock the scale_to_snr method to return a tensor of size 8 (batch size)
    mock_criterion.scale_to_snr.return_value = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0])
    
    # Mock the dataset to have SNR labels
    mock_dataloader.dataset.snr_labels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    mock_dataloader.dataset.dataset.snr_labels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    
    # Mock the batch size and labels
    mock_dataloader.batch_size = 8
    mock_dataloader.__iter__.return_value = [
        (
            torch.randn(8, 1, 32, 32),  # inputs
            torch.randint(0, 4, (8,)),   # modulation labels
            torch.randint(0, 5, (8,))    # SNR labels
        )
    ]
    
    # Mock dataset length to match batch size
    mock_dataloader.dataset.__len__.return_value = 8
    mock_dataloader.dataset.dataset.__len__.return_value = 8
    
    # Create a proper mock optimizer with param_groups
    mock_optimizer = MagicMock()
    mock_optimizer.param_groups = [{'lr': 0.001}]  # Add proper param_groups structure
    
    # Call the training function with mock objects
    best_model_path = train(
        model=mock_model,
        device=torch.device('cpu'),
        criterion_modulation=mock_criterion,
        criterion_snr=mock_criterion,
        criterion_dynamic=mock_criterion_dynamic,
        optimizer=mock_optimizer,
        scheduler=MagicMock(),
        train_loader=mock_dataloader,
        val_loader=mock_dataloader,
        epochs=2,
        save_dir="checkpoints"
    )
    
    # Capture only essential information about the mock calls
    training_state = {
        'train_called': mock_train.called,
        'best_model_path': best_model_path,
        'call_args': {
            'args': [arg.__class__.__name__ for arg in mock_train.call_args.args],
            'kwargs': list(mock_train.call_args.kwargs.keys())
        } if mock_train.called else None
    }
    
    # Save snapshot
    snapshot_path = os.path.join(snapshot_dir, "training_process_snapshot.json")
    save_model_snapshot(training_state, snapshot_path)
    
    # Verify snapshot was created
    assert os.path.exists(snapshot_path), f"Snapshot file not created at {snapshot_path}"


@pytest.mark.snapshot
def test_confusion_matrix_snapshot(mock_model, mock_dataloader, snapshot_dir):
    """Test creation and snapshot of confusion matrices."""
    # Create random true and predicted labels
    np.random.seed(42)
    
    # Use 100 samples, 4 mod classes, 5 SNR classes
    true_mod = np.random.randint(0, 4, 100)
    pred_mod = np.random.randint(0, 4, 100)
    true_snr = np.random.randint(0, 5, 100)
    pred_snr = np.random.randint(0, 5, 100)
    
    # Calculate confusion matrices
    from sklearn.metrics import confusion_matrix
    mod_cm = confusion_matrix(true_mod, pred_mod, labels=range(4))
    snr_cm = confusion_matrix(true_snr, pred_snr, labels=range(5))
    
    # Save snapshot
    confusion_data = {
        'mod_confusion_matrix': mod_cm,
        'snr_confusion_matrix': snr_cm,
        'mod_accuracy': np.sum(np.diag(mod_cm)) / np.sum(mod_cm) * 100,
        'snr_accuracy': np.sum(np.diag(snr_cm)) / np.sum(snr_cm) * 100
    }
    
    snapshot_path = os.path.join(snapshot_dir, "confusion_matrix_snapshot.json")
    save_model_snapshot(confusion_data, snapshot_path)
    
    # Verify snapshot was created
    assert os.path.exists(snapshot_path), f"Snapshot file not created at {snapshot_path}"
    print(f"Created confusion matrix snapshot at {snapshot_path}") 