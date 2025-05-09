import torch
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Callable, Union


def verify_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> bool:
    """
    Verify that a tensor has the expected shape.
    
    Args:
        tensor: The tensor to check
        expected_shape: The expected shape as a tuple
        
    Returns:
        True if shapes match, False otherwise
    """
    actual_shape = tensor.shape
    return actual_shape == expected_shape


def verify_function_signature(func: Callable, expected_args: List[str]) -> bool:
    """
    Verify that a function has the expected signature.
    
    Args:
        func: The function to check
        expected_args: List of expected argument names
        
    Returns:
        True if signature matches, False otherwise
    """
    import inspect
    sig = inspect.signature(func)
    actual_args = list(sig.parameters.keys())
    return set(actual_args) == set(expected_args)


def capture_model_outputs(model: torch.nn.Module, input_batch: torch.Tensor) -> Dict[str, Any]:
    """
    Capture the outputs of a model for a given input batch.
    
    Args:
        model: The PyTorch model
        input_batch: The input tensor
        
    Returns:
        Dictionary of model outputs, e.g. {'modulation': tensor, 'snr': tensor}
    """
    with torch.no_grad():
        modulation_output, snr_output = model(input_batch)
    
    return {
        'modulation': modulation_output.cpu().numpy(),
        'snr': snr_output.cpu().numpy()
    }


def save_model_snapshot(model_outputs: Dict[str, Any], file_path: str) -> None:
    """
    Save a snapshot of model outputs to a file for later comparison.
    
    Args:
        model_outputs: Dictionary of model outputs
        file_path: Path to save the snapshot
    """
    # Convert tensor outputs to numpy arrays and then to lists for JSON serialization
    json_outputs = {}
    for key, value in model_outputs.items():
        if isinstance(value, torch.Tensor):
            # Use detach() to handle tensors with gradients
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            json_outputs[key] = {
                'shape': value.shape,
                # Only save a subset of values to keep file size reasonable
                'values': value.flatten()[:100].tolist(),
                'mean': float(np.mean(value)),
                'std': float(np.std(value)),
                'min': float(np.min(value)),
                'max': float(np.max(value))
            }
        else:
            json_outputs[key] = value

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(json_outputs, f, indent=2)


def compare_model_snapshots(snapshot1: str, snapshot2: str, rtol: float = 1e-3) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two model snapshots for consistency.
    
    Args:
        snapshot1: Path to first snapshot
        snapshot2: Path to second snapshot
        rtol: Relative tolerance for float comparisons
        
    Returns:
        Tuple of (match_result, differences)
    """
    # Load snapshots
    with open(snapshot1, 'r') as f:
        data1 = json.load(f)
    
    with open(snapshot2, 'r') as f:
        data2 = json.load(f)
    
    # Check for matching keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    if keys1 != keys2:
        return False, {'missing_keys': list(keys1 - keys2), 'extra_keys': list(keys2 - keys1)}
    
    # Compare values
    differences = {}
    match_result = True
    
    for key in keys1:
        # Check if the values are dictionaries with tensor data
        if isinstance(data1[key], dict) and 'shape' in data1[key]:
            # Compare shapes
            if data1[key]['shape'] != data2[key]['shape']:
                match_result = False
                differences[f"{key}_shape"] = {
                    'snapshot1': data1[key]['shape'],
                    'snapshot2': data2[key]['shape']
                }
            
            # Compare statistics
            for stat in ['mean', 'std', 'min', 'max']:
                if abs(data1[key][stat] - data2[key][stat]) > rtol * abs(data1[key][stat]):
                    match_result = False
                    differences[f"{key}_{stat}"] = {
                        'snapshot1': data1[key][stat],
                        'snapshot2': data2[key][stat],
                        'difference': abs(data1[key][stat] - data2[key][stat])
                    }
        else:
            # Direct comparison for non-tensor data
            if data1[key] != data2[key]:
                match_result = False
                differences[key] = {
                    'snapshot1': data1[key],
                    'snapshot2': data2[key]
                }
    
    return match_result, differences


def create_mock_snr_mapping() -> Dict[int, float]:
    """Create a mock SNR index to value mapping"""
    return {i: i*5-20 for i in range(5)}  # -20, -15, -10, -5, 0


def generate_confusion_matrix(num_classes: int, accuracy: float = 0.7) -> np.ndarray:
    """
    Generate a mock confusion matrix with specified accuracy.
    
    Args:
        num_classes: Number of classes
        accuracy: Target accuracy on the diagonal
        
    Returns:
        NumPy array with confusion matrix
    """
    # Create a matrix with most predictions on the diagonal
    cm = np.zeros((num_classes, num_classes))
    
    # Set diagonal elements to reflect the target accuracy
    for i in range(num_classes):
        cm[i, i] = accuracy * 100  # Assuming 100 samples per class
        
        # Distribute remaining predictions randomly
        remaining = 100 - cm[i, i]
        for j in range(num_classes):
            if j != i:
                cm[i, j] = remaining / (num_classes - 1)
    
    return cm 