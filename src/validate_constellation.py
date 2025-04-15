# src/validate_constellation.py

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast
from contextlib import nullcontext
import os


def get_dataset_property(dataset_obj, property_name):
    """
    Helper function to access dataset properties consistently regardless of structure.
    
    Args:
        dataset_obj: The dataset object (could be Subset or direct dataset)
        property_name: The property to access
    
    Returns:
        The requested property
    """
    if hasattr(dataset_obj, 'dataset'):
        # Regular case with dataset_obj being a Subset
        return getattr(dataset_obj.dataset, property_name)
    else:
        # Curriculum case with dataset_obj being a direct dataset
        return getattr(dataset_obj, property_name)


def validate(model, device, criterion_modulation, criterion_snr, criterion_dynamic=None, val_loader=None, 
            use_autocast=False, use_curriculum=False, current_snr_list=None, use_snr_buckets=False, **kwargs):
    """
    Validate the model on the validation set.
    Returns validation loss, accuracies, and predictions for plotting.
    
    Args:
        model: Model to validate
        device: Device to validate on
        criterion_modulation: Modulation loss function
        criterion_snr: SNR loss function
        criterion_dynamic: Dynamic loss function (optional)
        val_loader: Validation data loader
        use_autocast: Whether to use automatic mixed precision
        use_curriculum: Whether curriculum learning is enabled
        current_snr_list: Current list of SNR values for curriculum
        use_snr_buckets: Whether SNR buckets are used
        **kwargs: Additional arguments for backward compatibility
    
    Returns:
        Tuple or Dict containing validation metrics and predictions, depending on the call pattern
    """
    model.eval()
    val_loss = 0.0
    modulation_loss_total = 0.0
    snr_loss_total = 0.0
    correct_modulation = 0
    correct_snr = 0
    total = 0
    correct_combined = 0  # Both modulation and SNR correct

    # Lists to store predictions and true labels for plotting
    all_pred_modulation_labels = []
    all_true_modulation_labels = []
    all_pred_snr_indices = []  # Store indices instead of values
    all_true_snr_indices = []  # Store indices instead of values
    
    # For curriculum learning, track per-SNR accuracy
    snr_correct_per_class = {}
    snr_total_per_class = {}
    
    # Get SNR values from dataset using the helper function
    dataset_obj = val_loader.dataset
    snr_labels_dict = get_dataset_property(dataset_obj, 'snr_labels')
    snr_values = list(snr_labels_dict.keys())
    
    # For direct confusion matrix calculation in curriculum mode
    if use_curriculum and current_snr_list:
        # Map SNR values to their curriculum positions
        curriculum_snr_values = sorted(current_snr_list)
        snr_value_to_position = {val: idx for idx, val in enumerate(curriculum_snr_values)}
        
        # Initialize confusion matrix for curriculum SNRs
        curriculum_cm = np.zeros((len(curriculum_snr_values), len(curriculum_snr_values)))
        
        # Store raw predictions and labels as actual SNR values
        raw_true_snr_values = []
        raw_pred_snr_values = []
    
    # Initialize SNR class counters if in curriculum mode
    if use_curriculum and current_snr_list:
        for snr in current_snr_list:
            # Find the index for this SNR value
            try:
                snr_idx = list(snr_labels_dict.keys()).index(snr)
                snr_correct_per_class[snr_idx] = 0
                snr_total_per_class[snr_idx] = 0
            except ValueError:
                print(f"Warning: SNR value {snr} not found in dataset labels")

    with torch.no_grad():
        for inputs, modulation_labels, snr_labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            modulation_labels = modulation_labels.to(device)
            snr_labels = snr_labels.to(device)

            # Use autocast for mixed precision if requested
            context = autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu') if use_autocast else nullcontext()
            
            with context:
                # Only pass inputs to the model
                modulation_output, snr_output = model(inputs)
                
                # Calculate losses using criterion functions
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)
                
                # If criterion_dynamic is provided, use it to combine losses
                if criterion_dynamic is not None:
                    # Pass losses list to criterion_dynamic
                    loss, _ = criterion_dynamic([loss_modulation, loss_snr])
                else:
                    # Default behavior if criterion_dynamic is not provided
                    loss = loss_modulation + loss_snr

            val_loss += loss.item()
            modulation_loss_total += loss_modulation.item()
            snr_loss_total += loss_snr.item()

            # Get predicted classes
            _, predicted_modulation = modulation_output.max(1)
            _, predicted_snr = snr_output.max(1)
            
            total += modulation_labels.size(0)
            correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
            correct_snr += predicted_snr.eq(snr_labels).sum().item()
            
            # For curriculum learning, track per-SNR class accuracy
            if use_curriculum and current_snr_list:
                for i in range(len(snr_labels)):
                    try:
                        # Convert to actual SNR values
                        true_idx = int(snr_labels[i].item())
                        pred_idx = int(predicted_snr[i].item())
                        
                        if true_idx < len(snr_values):
                            true_snr_value = snr_values[true_idx]
                            
                            # Only count if this SNR is in our curriculum
                            if true_snr_value in current_snr_list:
                                # Track per-class metrics
                                if true_idx in snr_correct_per_class:
                                    snr_total_per_class[true_idx] += 1
                                    if pred_idx == true_idx:
                                        snr_correct_per_class[true_idx] += 1
                                
                                # For direct confusion matrix calculation
                                if pred_idx < len(snr_values):
                                    pred_snr_value = snr_values[pred_idx]
                                    
                                    # Store raw SNR values
                                    raw_true_snr_values.append(true_snr_value)
                                    raw_pred_snr_values.append(pred_snr_value)
                                    
                                    # Update curriculum confusion matrix if both values are in curriculum
                                    if true_snr_value in snr_value_to_position and pred_snr_value in snr_value_to_position:
                                        true_pos = snr_value_to_position[true_snr_value]
                                        pred_pos = snr_value_to_position[pred_snr_value]
                                        curriculum_cm[true_pos, pred_pos] += 1
                    except (IndexError, ValueError) as e:
                        continue

            # Store predictions and true labels
            all_pred_modulation_labels.extend(predicted_modulation.cpu().numpy())
            all_true_modulation_labels.extend(modulation_labels.cpu().numpy())
            all_pred_snr_indices.extend(predicted_snr.cpu().numpy())  # Store indices
            all_true_snr_indices.extend(snr_labels.cpu().numpy())    # Store indices

    # Calculate average loss and accuracies
    val_loss = val_loss / len(val_loader)
    modulation_loss_total = modulation_loss_total / len(val_loader)
    snr_loss_total = snr_loss_total / len(val_loader)
    val_modulation_accuracy = 100.0 * correct_modulation / total
    val_snr_accuracy = 100.0 * correct_snr / total
    
    # Calculate SNR MAE (Mean Absolute Error) if possible
    val_snr_mae = 0.0
    try:
        if hasattr(dataset_obj, 'get_actual_snr_values') or hasattr(dataset_obj, 'dataset') and hasattr(dataset_obj.dataset, 'get_actual_snr_values'):
            # Calculate SNR MAE using predicted indices and actual SNR values
            true_snr_values_for_mae = []
            pred_snr_values_for_mae = []
            
            # Get the function to convert indices to actual SNR values
            get_snr_func = None
            if hasattr(dataset_obj, 'get_actual_snr_values'):
                get_snr_func = dataset_obj.get_actual_snr_values
            elif hasattr(dataset_obj, 'dataset') and hasattr(dataset_obj.dataset, 'get_actual_snr_values'):
                get_snr_func = dataset_obj.dataset.get_actual_snr_values
                
            if get_snr_func:
                success_count = 0
                error_count = 0
                for i in range(len(all_true_snr_indices)):
                    try:
                        true_idx = int(all_true_snr_indices[i])
                        pred_idx = int(all_pred_snr_indices[i])
                        
                        true_val = get_snr_func(true_idx)
                        pred_val = get_snr_func(pred_idx)
                        
                        true_snr_values_for_mae.append(true_val)
                        pred_snr_values_for_mae.append(pred_val)
                        success_count += 1
                    except (IndexError, ValueError, TypeError) as e:
                        error_count += 1
                        # Skip this sample for MAE calculation
                        continue
                
                # Calculate MAE if we have successful conversions
                if true_snr_values_for_mae:
                    abs_errors = [abs(true - pred) for true, pred in zip(true_snr_values_for_mae, pred_snr_values_for_mae)]
                    val_snr_mae = sum(abs_errors) / len(abs_errors)
                    
                    # Report any errors that occurred
                    if error_count > 0:
                        percent_errors = (error_count / (success_count + error_count)) * 100
                        print(f"Warning: {error_count} SNR value conversion errors ({percent_errors:.1f}%) during MAE calculation")
                else:
                    print("Warning: No valid SNR conversions for MAE calculation")
                    val_snr_mae = 0.0
        else:
            # Fallback: use indices directly as a proxy for calculating MAE
            abs_diffs = np.abs(np.array(all_true_snr_indices) - np.array(all_pred_snr_indices))
            val_snr_mae = np.mean(abs_diffs) if len(abs_diffs) > 0 else 0.0
    except Exception as e:
        print(f"Warning: Could not calculate SNR MAE: {str(e)}")
        val_snr_mae = 0.0
    
    # Calculate per-SNR class accuracy for curriculum learning
    snr_class_accuracies = {}
    inverse_snr_labels = get_dataset_property(dataset_obj, 'inverse_snr_labels')
    
    if use_curriculum and current_snr_list:
        for snr_idx in snr_correct_per_class:
            if snr_total_per_class[snr_idx] > 0:
                accuracy = 100.0 * snr_correct_per_class[snr_idx] / snr_total_per_class[snr_idx]
                snr_value = inverse_snr_labels[snr_idx]
                snr_class_accuracies[f"snr_{snr_value}db_accuracy"] = accuracy
    
    # Prepare and return validation metrics
    metrics = {
        'val_loss': val_loss,
        'val_modulation_loss': modulation_loss_total,
        'val_snr_loss': snr_loss_total,
        'val_modulation_accuracy': val_modulation_accuracy,
        'val_snr_accuracy': val_snr_accuracy,
        'val_snr_mae': val_snr_mae,  # Add SNR MAE to metrics
        'true_modulation_labels': all_true_modulation_labels,
        'pred_modulation_labels': all_pred_modulation_labels,
        'true_snr_indices': all_true_snr_indices,
        'pred_snr_indices': all_pred_snr_indices,
    }
    
    # Calculate combined accuracy (both modulation and SNR correct)
    correct_combined = 0
    for i in range(len(all_true_modulation_labels)):
        if (all_true_modulation_labels[i] == all_pred_modulation_labels[i] and
            all_true_snr_indices[i] == all_pred_snr_indices[i]):
            correct_combined += 1
    combined_accuracy = 100.0 * correct_combined / total if total > 0 else 0.0
    
    # Add combined accuracy to metrics
    metrics['val_combined_accuracy'] = combined_accuracy
    
    # Add per-SNR metrics if in curriculum mode
    if use_curriculum and current_snr_list:
        metrics.update(snr_class_accuracies)
        
        # Add custom confusion matrix for curriculum mode
        metrics['curriculum_cm'] = curriculum_cm
        metrics['curriculum_snr_values'] = curriculum_snr_values
        metrics['raw_true_snr_values'] = raw_true_snr_values
        metrics['raw_pred_snr_values'] = raw_pred_snr_values
        
        # Calculate average accuracy across current curriculum SNRs
        curriculum_snr_accuracies = [acc for k, acc in snr_class_accuracies.items()]
        if curriculum_snr_accuracies:
            metrics['curriculum_snr_avg_accuracy'] = sum(curriculum_snr_accuracies) / len(curriculum_snr_accuracies)
    
    # For backward compatibility with test_constellation.py, return tuple format if criterion_dynamic is None
    if criterion_dynamic is None:
        return (val_loss, modulation_loss_total, snr_loss_total, val_modulation_accuracy, 
                val_snr_accuracy, combined_accuracy, 
                np.array(all_true_modulation_labels), np.array(all_pred_modulation_labels),
                np.array(all_true_snr_indices), np.array(all_pred_snr_indices))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, classes, title=None, normalize=False, save_path=None, cm=None):
    """
    Plot a confusion matrix using seaborn's heatmap.
    
    Args:
        y_true: True labels (if cm is None)
        y_pred: Predicted labels (if cm is None)
        classes: List of class names
        title: Title for the plot
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
        cm: Pre-calculated confusion matrix (optional)
    """
    if cm is None and y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm / row_sums
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
    
    # Calculate accuracy if we have the raw confusion matrix
    if not normalize and cm is not None:
        diag_sum = np.sum(np.diag(cm))
        total = np.sum(cm)
        acc = diag_sum / total * 100 if total > 0 else 0
        title = f"{title}\nAccuracy: {acc:.2f}%" if title else f"Confusion Matrix\nAccuracy: {acc:.2f}%"
    
    if title:
        plt.title(title)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_validation_confusion_matrices(true_modulation, pred_modulation, true_snr_indices, pred_snr_indices, 
                                       mod_classes=None, save_dir=None, epoch=None, use_curriculum=False, 
                                       current_snr_list=None, metrics=None):
    """
    Plot confusion matrices for modulation and SNR predictions
    """
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(true_modulation, torch.Tensor):
        true_modulation = true_modulation.cpu().numpy()
    if isinstance(pred_modulation, torch.Tensor):
        pred_modulation = pred_modulation.cpu().numpy()
    if isinstance(true_snr_indices, torch.Tensor):
        true_snr_indices = true_snr_indices.cpu().numpy()
    if isinstance(pred_snr_indices, torch.Tensor):
        pred_snr_indices = pred_snr_indices.cpu().numpy()
    
    # Convert to numpy arrays if they're lists
    true_modulation = np.array(true_modulation)
    pred_modulation = np.array(pred_modulation)
    true_snr_indices = np.array(true_snr_indices)
    pred_snr_indices = np.array(pred_snr_indices)
    
    # Ensure all inputs have the same length
    min_length = min(len(true_modulation), len(pred_modulation), 
                     len(true_snr_indices), len(pred_snr_indices))
    if min_length < len(true_modulation):
        print(f"WARNING: Truncating arrays from {len(true_modulation)} to {min_length} elements")
        true_modulation = true_modulation[:min_length]
        pred_modulation = pred_modulation[:min_length]
        true_snr_indices = true_snr_indices[:min_length]
        pred_snr_indices = pred_snr_indices[:min_length]
    
    # Rename input parameters for clarity (accepting both naming conventions)
    true_mod_labels = true_modulation
    pred_mod_labels = pred_modulation
    
    print(f"Plot confusion matrices with shape: true_mod={true_mod_labels.shape}, pred_mod={pred_mod_labels.shape}")
    print(f"true_snr={true_snr_indices.shape}, pred_snr={pred_snr_indices.shape}")
    print(f"true_snr_indices dtype: {true_snr_indices.dtype}, pred_snr_indices dtype: {pred_snr_indices.dtype}")
    print(f"true_snr_indices unique values: {np.unique(true_snr_indices)}")
    print(f"pred_snr_indices unique values: {np.unique(pred_snr_indices)}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up classes for modulation
    if mod_classes is None:
        mod_classes = [str(i) for i in range(np.max(true_modulation) + 1)]
    
    # Plot modulation confusion matrix
    plot_confusion_matrix(true_mod_labels, pred_mod_labels, classes=mod_classes, 
                          title=f'Modulation Confusion Matrix (Epoch {epoch})' if epoch else 'Modulation Confusion Matrix',
                          normalize=True, save_path=os.path.join(save_dir, f'mod_conf_matrix_epoch_{epoch}.png') if epoch else None)
    
    # Handle SNR confusion matrix
    # Define standard SNR values if not provided
    if current_snr_list is None:
        current_snr_list = [-20, -10, 10, 30]  # Default curriculum SNR values
        print(f"Using default SNR list: {current_snr_list}")
    else:
        print(f"Using provided SNR list: {current_snr_list}")
    
    # Determine if values are indices or actual SNR values
    # More reliable detection based on value ranges and uniqueness
    unique_true_snrs = np.unique(true_snr_indices)
    
    # If there are only a few unique values (matching our curriculum)
    # and they're small integers, they're likely indices
    is_indices = (len(unique_true_snrs) <= len(current_snr_list) and 
                  np.all(unique_true_snrs >= 0) and 
                  np.all(unique_true_snrs < len(current_snr_list)))
    
    print(f"Detection - SNR values are {'indices' if is_indices else 'actual values'}")
    
    # Map indices to actual SNR values if needed
    if is_indices:
        print(f"Mapping indices to SNR values using curriculum: {current_snr_list}")
        # Convert indices to integers to ensure proper indexing
        true_indices = true_snr_indices.astype(int)
        pred_indices = pred_snr_indices.astype(int)
        
        # Map indices to values, handling out-of-bounds indices
        true_snr_values = np.zeros_like(true_indices, dtype=float)
        pred_snr_values = np.zeros_like(pred_indices, dtype=float)
        
        # Create masks for valid indices
        true_valid_mask = (true_indices >= 0) & (true_indices < len(current_snr_list))
        pred_valid_mask = (pred_indices >= 0) & (pred_indices < len(current_snr_list))
        
        print(f"Valid true indices range: {np.min(true_indices) if len(true_indices) > 0 else 'N/A'} to {np.max(true_indices) if len(true_indices) > 0 else 'N/A'}")
        print(f"Valid pred indices range: {np.min(pred_indices) if len(pred_indices) > 0 else 'N/A'} to {np.max(pred_indices) if len(pred_indices) > 0 else 'N/A'}")
        print(f"Current SNR list length: {len(current_snr_list)}")
        
        # Apply mapping only for valid indices
        for i, snr_value in enumerate(current_snr_list):
            true_snr_values[true_indices == i] = snr_value
            pred_snr_values[pred_indices == i] = snr_value
        
        # Handle invalid indices by setting to a safe default (e.g., the first SNR value)
        if len(current_snr_list) > 0:
            default_snr = current_snr_list[0]
            invalid_count_true = np.sum(~true_valid_mask)
            invalid_count_pred = np.sum(~pred_valid_mask)
            
            if invalid_count_true > 0 or invalid_count_pred > 0:
                print(f"WARNING: Found {invalid_count_true} invalid true indices and {invalid_count_pred} invalid pred indices")
                print(f"Setting invalid indices to default SNR value: {default_snr}")
            
            true_snr_values[~true_valid_mask] = default_snr
            pred_snr_values[~pred_valid_mask] = default_snr
            
        print(f"Valid true indices: {np.sum(true_valid_mask)}/{len(true_indices)}")
        print(f"Valid pred indices: {np.sum(pred_valid_mask)}/{len(pred_indices)}")
    else:
        # Values are already SNR values
        true_snr_values = true_snr_indices
        pred_snr_values = pred_snr_indices
    
    print(f"SNR values after mapping - true: {np.unique(true_snr_values)}")
    print(f"SNR values after mapping - pred: {np.unique(pred_snr_values)}")
    
    # Create SNR class labels from unique values or curriculum
    if use_curriculum:
        snr_classes = [str(int(snr)) for snr in current_snr_list]
        
        # Create a custom confusion matrix to ensure all curriculum SNRs are represented
        snr_cm = np.zeros((len(current_snr_list), len(current_snr_list)))
        
        # Create lookup for faster matching
        snr_value_to_idx = {snr: i for i, snr in enumerate(current_snr_list)}
        
        # Map values to indices in the confusion matrix
        for i in range(len(true_snr_values)):
            true_val = true_snr_values[i]
            pred_val = pred_snr_values[i]
            
            # Find the closest SNR values in our curriculum list
            true_idx = -1
            pred_idx = -1
            
            # Direct match first (most efficient)
            if true_val in snr_value_to_idx:
                true_idx = snr_value_to_idx[true_val]
            else:
                # Find closest match
                true_diffs = [abs(true_val - snr) for snr in current_snr_list]
                if true_diffs:
                    true_idx = np.argmin(true_diffs)
            
            if pred_val in snr_value_to_idx:
                pred_idx = snr_value_to_idx[pred_val]
            else:
                # Find closest match
                pred_diffs = [abs(pred_val - snr) for snr in current_snr_list]
                if pred_diffs:
                    pred_idx = np.argmin(pred_diffs)
            
            # Only update if we found valid indices
            if true_idx >= 0 and pred_idx >= 0:
                snr_cm[true_idx, pred_idx] += 1
        
        print(f"Created custom SNR confusion matrix with shape: {snr_cm.shape}")
        print(f"Confusion matrix contains {np.sum(snr_cm)} samples")
        
        # Check if the confusion matrix is valid
        if np.sum(snr_cm) > 0:
            # Plot SNR confusion matrix with the custom matrix
            plot_confusion_matrix(None, None, classes=snr_classes, 
                                title=f'SNR Confusion Matrix (Epoch {epoch})' if epoch else 'SNR Confusion Matrix',
                                normalize=True, 
                                save_path=os.path.join(save_dir, f'snr_conf_matrix_epoch_{epoch}.png') if epoch else None,
                                cm=snr_cm)
        else:
            print("WARNING: No valid samples for SNR confusion matrix, falling back to standard method")
            # Fall back to standard method
            use_curriculum = False
    else:
        # Use automatic confusion matrix from sklearn
        unique_snrs = sorted(list(set(np.concatenate([np.unique(true_snr_values), np.unique(pred_snr_values)]))))
        
        # Filter out any potential NaN or inf values
        unique_snrs = [snr for snr in unique_snrs if np.isfinite(snr)]
        
        # Make sure we have valid SNR values
        if len(unique_snrs) == 0:
            print("WARNING: No valid SNR values found, using default SNR classes")
            unique_snrs = [-20, -10, 0, 10, 20, 30]  # Default fallback values
        
        # Convert to string labels and ensure they're unique
        snr_classes = [str(int(snr)) for snr in unique_snrs]
        
        # Print the SNR class information
        print(f"Using SNR classes: {snr_classes}")
        print(f"True SNR values shape: {true_snr_values.shape}")
        print(f"Pred SNR values shape: {pred_snr_values.shape}")
        
        try:
            # Attempt to create the confusion matrix
            cm = confusion_matrix(true_snr_values, pred_snr_values, labels=unique_snrs)
            
            # Plot SNR confusion matrix
            plot_confusion_matrix(None, None, classes=snr_classes, 
                                title=f'SNR Confusion Matrix (Epoch {epoch})' if epoch else 'SNR Confusion Matrix',
                                normalize=True, 
                                save_path=os.path.join(save_dir, f'snr_conf_matrix_epoch_{epoch}.png') if epoch else None,
                                cm=cm)
        except Exception as e:
            print(f"ERROR creating SNR confusion matrix: {str(e)}")
            print("Attempting simplified confusion matrix...")
            
            # Try a simpler approach: just use the indices directly
            try:
                simple_cm = confusion_matrix(true_snr_indices, pred_snr_indices)
                simple_classes = [str(i) for i in range(simple_cm.shape[0])]
                
                plot_confusion_matrix(None, None, classes=simple_classes, 
                                    title=f'SNR Index Confusion Matrix (Epoch {epoch})' if epoch else 'SNR Index Confusion Matrix',
                                    normalize=True, 
                                    save_path=os.path.join(save_dir, f'snr_index_conf_matrix_epoch_{epoch}.png') if epoch else None,
                                    cm=simple_cm)
            except Exception as e2:
                print(f"ERROR creating simplified SNR confusion matrix: {str(e2)}")
                print("Unable to create SNR confusion matrix.")
