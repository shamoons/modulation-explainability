import pytest
import torch
import sys
import os
import inspect
from unittest.mock import MagicMock, patch

# Add parent directory to path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import function to test
from src.validate_constellation import validate, plot_validation_confusion_matrices
from tests.utils.test_helpers import verify_tensor_shape, verify_function_signature


@pytest.mark.contract
def test_validate_function_signature():
    """Test that the validate function has the expected signature."""
    expected_params = [
        'model', 'device', 'criterion_modulation', 'criterion_snr', 
        'criterion_dynamic', 'val_loader', 'use_autocast', 
        'use_curriculum', 'current_snr_list', 'use_snr_buckets'
    ]
    
    # Get the actual parameters
    sig = inspect.signature(validate)
    actual_params = list(sig.parameters.keys())
    
    # Check that all expected parameters exist
    for param in expected_params:
        assert param in actual_params, f"Expected parameter '{param}' not found in validate function signature"


@pytest.mark.contract
def test_validate_return_types(mock_model, mock_device, mock_criterion, mock_dataloader):
    """Test that the validate function returns the expected types."""
    # Call with criterion_dynamic=None to get tuple return
    result = validate(
        model=mock_model,
        device=mock_device,
        criterion_modulation=mock_criterion,
        criterion_snr=mock_criterion,
        criterion_dynamic=None,
        val_loader=mock_dataloader
    )
    
    # Check that result is a tuple of the right length
    assert isinstance(result, tuple), "validate should return a tuple when criterion_dynamic is None"
    assert len(result) == 10, "validate should return a 10-tuple when criterion_dynamic is None"
    
    # Check individual return types
    val_loss, mod_loss, snr_loss, mod_acc, snr_acc, combined_acc, true_mod, pred_mod, true_snr, pred_snr = result
    
    assert isinstance(val_loss, float), "val_loss should be a float"
    assert isinstance(mod_loss, float), "mod_loss should be a float"
    assert isinstance(snr_loss, float), "snr_loss should be a float"
    assert isinstance(mod_acc, float), "mod_acc should be a float"
    assert isinstance(snr_acc, float), "snr_acc should be a float"
    assert isinstance(combined_acc, float), "combined_acc should be a float"
    assert isinstance(true_mod, (torch.Tensor, type(torch.tensor(1).numpy()))), "true_mod should be a tensor or numpy array"
    assert isinstance(pred_mod, (torch.Tensor, type(torch.tensor(1).numpy()))), "pred_mod should be a tensor or numpy array"
    assert isinstance(true_snr, (torch.Tensor, type(torch.tensor(1).numpy()))), "true_snr should be a tensor or numpy array"
    assert isinstance(pred_snr, (torch.Tensor, type(torch.tensor(1).numpy()))), "pred_snr should be a tensor or numpy array"


@pytest.mark.contract
def test_validate_with_criterion_dynamic(mock_model, mock_device, mock_criterion, mock_criterion_dynamic, mock_dataloader):
    """Test that validate returns a dict when criterion_dynamic is provided."""
    result = validate(
        model=mock_model,
        device=mock_device,
        criterion_modulation=mock_criterion,
        criterion_snr=mock_criterion,
        criterion_dynamic=mock_criterion_dynamic,
        val_loader=mock_dataloader
    )
    
    # Check result is a dict with expected keys
    assert isinstance(result, dict), "validate should return a dict when criterion_dynamic is provided"
    
    expected_keys = [
        'val_loss', 'val_modulation_loss', 'val_snr_loss',
        'val_modulation_accuracy', 'val_snr_accuracy', 'val_snr_mae',
        'true_modulation_labels', 'pred_modulation_labels',
        'true_snr_indices', 'pred_snr_indices', 'val_combined_accuracy'
    ]
    
    for key in expected_keys:
        assert key in result, f"Expected key '{key}' not found in result dict"


@pytest.mark.contract
def test_confusion_matrix_function_signature():
    """Test that the plot_validation_confusion_matrices function has the expected signature."""
    expected_params = [
        'true_modulation', 'pred_modulation', 'true_snr_indices', 'pred_snr_indices',
        'mod_classes', 'save_dir', 'epoch', 'use_curriculum', 'current_snr_list', 'metrics'
    ]
    
    # Get the actual parameters
    sig = inspect.signature(plot_validation_confusion_matrices)
    actual_params = list(sig.parameters.keys())
    
    # Check that all expected parameters exist
    for param in expected_params:
        assert param in actual_params, f"Expected parameter '{param}' not found in plot_validation_confusion_matrices function signature"


@pytest.mark.contract
def test_tensor_processing_with_mocks(mock_model, test_input_batch, test_labels):
    """Test tensor shapes are processed correctly through the model."""
    # Get model outputs
    mod_labels, snr_labels = test_labels
    mod_output, snr_output = mock_model(test_input_batch)
    
    # Check shapes
    assert verify_tensor_shape(mod_output, (test_input_batch.size(0), 4)), "Modulation output should have shape (batch_size, num_classes)"
    assert verify_tensor_shape(snr_output, (test_input_batch.size(0), 5)), "SNR output should have shape (batch_size, num_snr_classes)"
    
    # Check tensor types
    assert mod_output.dtype == torch.float32, "Modulation output should be float32"
    assert snr_output.dtype == torch.float32, "SNR output should be float32" 