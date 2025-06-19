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


def validate_constellation(model, val_dataloader, criterion_modulation, criterion_snr, criterion_dynamic, device, 
                            use_curriculum=False, curriculum_manager=None, 
                            save_results_dir=None, epoch=None, 
                            visualize=False, mod_classes=None):
    """
    Validate a constellation recognition model
    
    Args:
        model: The PyTorch model to validate
        val_dataloader: Validation data loader
        criterion_modulation: Loss function for modulation classification
        criterion_snr: Loss function for SNR classification
        criterion_dynamic: Dynamic loss weighting function
        device: Device to run validation on
        use_curriculum: Whether to use curriculum learning
        curriculum_manager: CurriculumManager instance if using curriculum
        save_results_dir: Directory to save validation results
        epoch: Current epoch number
        visualize: Whether to visualize validation results
        mod_classes: List of modulation class names for visualization
    
    Returns:
        val_loss, mod_accuracy, snr_accuracy, metrics_dict
    """
    # Set model to evaluation mode
    model.eval()
    
    val_loss = 0.0
    correct_mod = 0
    correct_snr = 0
    total = 0
    
    # Store ground truth and predictions for visualization
    true_mod_labels = []
    pred_mod_labels = []
    true_snr_indices = []
    pred_snr_indices = []
    
    # For curriculum mode - track per-SNR accuracy and record a direct confusion matrix
    metrics = {}
    
    # Get the SNR values from the dataset for proper mapping
    try:
        if hasattr(val_dataloader.dataset, 'dataset'):
            dataset_snr_labels = val_dataloader.dataset.dataset.snr_labels
        else:
            dataset_snr_labels = val_dataloader.dataset.snr_labels
            
        snr_values = list(dataset_snr_labels.keys())
        metrics['snr_values'] = snr_values
        print(f"Available SNR values for validation: {snr_values}")
    except Exception as e:
        print(f"Error getting SNR values from dataset: {str(e)}")
        snr_values = []
        
    # Use curriculum SNR values if available
    if use_curriculum and curriculum_manager is not None:
        # Get current SNR values
        current_snr_list = curriculum_manager.get_current_snr_list()
        metrics['current_snr_list'] = current_snr_list
        print(f"Current curriculum stage: {curriculum_manager.current_stage}")
        print(f"Current SNR list: {current_snr_list}")
        print(f"Length of current SNR list: {len(current_snr_list)}")
        
        # Initialize confusion matrix for curriculum SNRs
        n_curr_classes = len(current_snr_list)
        curriculum_cm = np.zeros((n_curr_classes, n_curr_classes))
        
        # Store mapping from SNR values to their curriculum indices
        snr_to_curr_idx = {snr: i for i, snr in enumerate(sorted(current_snr_list))}
        metrics['curriculum_snr_values'] = sorted(current_snr_list)
        
    # Disable gradient computation for validation
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            inputs, mod_labels, snr_indices = \
                data[0].to(device), data[1].to(device), data[2].to(device)
            
            # Forward pass
            mod_out, snr_out = model(inputs)
            
            # Calculate individual losses
            mod_loss = criterion_modulation(mod_out, mod_labels)
            snr_loss = criterion_snr(snr_out, snr_indices)
            
            # Combine losses using dynamic weighting if available
            if criterion_dynamic is not None:
                loss, _ = criterion_dynamic([mod_loss, snr_loss])
            else:
                loss = mod_loss + snr_loss
            
            # Update validation loss
            val_loss += loss.item()
            
            # Get predictions
            _, mod_preds = torch.max(mod_out, 1)
            _, snr_preds = torch.max(snr_out, 1)
            
            # Update accuracy metrics
            batch_size = mod_labels.size(0)
            total += batch_size
            correct_mod += (mod_preds == mod_labels).sum().item()
            correct_snr += (snr_preds == snr_indices).sum().item()
            
            # Store for visualization
            true_mod_labels.extend(mod_labels.cpu().numpy())
            pred_mod_labels.extend(mod_preds.cpu().numpy())
            true_snr_indices.extend(snr_indices.cpu().numpy())
            pred_snr_indices.extend(snr_preds.cpu().numpy())
            
            # Track per-SNR accuracy for curriculum mode
            if use_curriculum and curriculum_manager is not None:
                # For each sample in batch, update the confusion matrix
                for i in range(batch_size):
                    try:
                        # Get the actual SNR value for this sample
                        if hasattr(val_dataloader.dataset, 'get_actual_snr_values'):
                            true_snr_idx = snr_indices[i].item()
                            pred_snr_idx = snr_preds[i].item()
                            
                            # Safely convert indices to SNR values with bounds checking
                            try:
                                true_snr_val = val_dataloader.dataset.get_actual_snr_values(true_snr_idx)
                                pred_snr_val = val_dataloader.dataset.get_actual_snr_values(pred_snr_idx)
                                
                                # Check if these SNRs are in our current curriculum
                                if true_snr_val in snr_to_curr_idx and pred_snr_val in snr_to_curr_idx:
                                    true_curr_idx = snr_to_curr_idx[true_snr_val]
                                    pred_curr_idx = snr_to_curr_idx[pred_snr_val]
                                    curriculum_cm[true_curr_idx, pred_curr_idx] += 1
                            except (IndexError, ValueError) as e:
                                # Silently continue if conversion fails
                                # We don't want to pollute logs with many errors
                                pass
                    except Exception as e:
                        # Catch broader exceptions but don't halt validation
                        continue
    
    # Calculate average metrics
    val_loss /= len(val_dataloader)
    mod_accuracy = 100 * correct_mod / total
    snr_accuracy = 100 * correct_snr / total
    
    # Store metrics for visualization
    metrics['mod_accuracy'] = mod_accuracy
    metrics['snr_accuracy'] = snr_accuracy
    
    if use_curriculum and curriculum_manager is not None:
        # Add curriculum metrics
        metrics['curriculum_cm'] = curriculum_cm
        print(f"\nDirect Curriculum SNR Confusion Matrix:")
        print(curriculum_cm)
        
        # Calculate overall accuracy from confusion matrix for verification
        cm_diagonal_sum = np.sum(np.diag(curriculum_cm))
        cm_total = np.sum(curriculum_cm)
        cm_accuracy = 100 * cm_diagonal_sum / cm_total if cm_total > 0 else 0
        print(f"Overall SNR accuracy from confusion matrix: {cm_accuracy:.2f}%")
        print(f"Overall SNR accuracy from predictions: {snr_accuracy:.2f}%")
        metrics['cm_snr_accuracy'] = cm_accuracy
    
    # Visualize validation results if requested
    if visualize and save_results_dir:
        # Create visualization directory if it doesn't exist
        os.makedirs(save_results_dir, exist_ok=True)
        
        # Plot confusion matrices
        from src.validate_constellation import plot_validation_confusion_matrices
        
        # Pass metric information for visualization
        plot_validation_confusion_matrices(
            true_modulation=true_mod_labels,
            pred_modulation=pred_mod_labels,
            true_snr_indices=true_snr_indices,
            pred_snr_indices=pred_snr_indices,
            mod_classes=mod_classes,
            save_dir=save_results_dir,
            epoch=epoch,
            use_curriculum=use_curriculum,
            current_snr_list=curriculum_manager.get_current_snr_list() if use_curriculum and curriculum_manager else None,
            metrics=metrics
        )
    
    return val_loss, mod_accuracy, snr_accuracy, metrics


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
    Plot confusion matrices for modulation and SNR predictions, handling both direct values and indices.
    
    Args:
        true_modulation (array): True modulation labels
        pred_modulation (array): Predicted modulation labels
        true_snr_indices (array): True SNR indices or values
        pred_snr_indices (array): Predicted SNR indices or values
        mod_classes (list): List of modulation class names
        save_dir (str): Directory to save plots
        epoch (int): Current epoch number
        use_curriculum (bool): Whether to use curriculum learning
        current_snr_list (list): List of current SNR values in curriculum
        metrics (dict): Additional metrics to use
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # Convert all inputs to numpy arrays if they aren't already
    true_mod_labels = np.array(true_modulation)
    pred_mod_labels = np.array(pred_modulation)
    true_snr_indices = np.array(true_snr_indices)
    pred_snr_indices = np.array(pred_snr_indices)
    
    # Check that all inputs have the same length
    if (len(true_mod_labels) != len(pred_mod_labels) or 
        len(true_snr_indices) != len(pred_snr_indices) or
        len(true_mod_labels) != len(true_snr_indices)):
        raise ValueError("All input arrays must have the same length")
    
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
        # Try to get SNR list from metrics if available
        if metrics and 'current_snr_list' in metrics:
            current_snr_list = metrics['current_snr_list']
        else:
            current_snr_list = [-20, -10, 10, 30]  # Default curriculum SNR values
        print(f"Using default SNR list: {current_snr_list}")
    else:
        print(f"Using provided SNR list: {current_snr_list}")
    
    # Make sure current_snr_list is sorted for consistent mapping
    current_snr_list = sorted(current_snr_list)
    
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
        
        # Handle invalid indices by setting to a safe default or filtering them out
        if len(current_snr_list) > 0:
            default_snr = current_snr_list[0]
            invalid_count_true = np.sum(~true_valid_mask)
            invalid_count_pred = np.sum(~pred_valid_mask)
            
            if invalid_count_true > 0 or invalid_count_pred > 0:
                print(f"WARNING: Found {invalid_count_true} invalid true indices and {invalid_count_pred} invalid pred indices")
                
                # Instead of setting them to default value, filter them out
                valid_samples = true_valid_mask & pred_valid_mask
                if np.sum(valid_samples) > 0:
                    print(f"Filtering out invalid indices, keeping {np.sum(valid_samples)}/{len(true_indices)} samples")
                    true_snr_values = true_snr_values[valid_samples]
                    pred_snr_values = pred_snr_values[valid_samples]
                else:
                    print(f"WARNING: No valid samples left after filtering. Setting invalid indices to default SNR value: {default_snr}")
                    # If we'd filter everything out, set to default as fallback
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
    
    # Check if we have a curriculum confusion matrix already calculated
    if use_curriculum and metrics and 'curriculum_cm' in metrics:
        print("Using pre-calculated curriculum confusion matrix from validation")
        curriculum_cm = metrics['curriculum_cm']
        
        if curriculum_cm.shape[0] == len(current_snr_list):
            snr_classes = [str(int(snr)) for snr in current_snr_list]
            
            # Plot SNR confusion matrix with the pre-calculated matrix
            plot_confusion_matrix(None, None, classes=snr_classes, 
                                 title=f'SNR Confusion Matrix (Epoch {epoch})' if epoch else 'SNR Confusion Matrix',
                                 normalize=True, 
                                 save_path=os.path.join(save_dir, f'snr_conf_matrix_epoch_{epoch}.png') if epoch else None,
                                 cm=curriculum_cm)
            return
        else:
            print(f"Warning: Pre-calculated confusion matrix shape {curriculum_cm.shape} doesn't match current SNR list length {len(current_snr_list)}")
    
    # Create SNR class labels from unique values or curriculum
    if use_curriculum:
        snr_classes = [str(int(snr)) for snr in current_snr_list]
        
        # Create a custom confusion matrix to ensure all curriculum SNRs are represented
        snr_cm = np.zeros((len(current_snr_list), len(current_snr_list)))
        
        # Create lookup for faster matching
        snr_value_to_idx = {snr: i for i, snr in enumerate(current_snr_list)}
        
        # Check if we actually have data to work with
        if len(true_snr_values) == 0 or len(pred_snr_values) == 0:
            print("WARNING: No valid SNR values found for confusion matrix")
            
            # Try a simplified approach with indices directly
            try:
                # Check if we can safely create a confusion matrix using indices
                max_idx = max(np.max(true_snr_indices), np.max(pred_snr_indices))
                if max_idx < 30:  # Reasonable max index for safety
                    simple_cm = confusion_matrix(true_snr_indices, pred_snr_indices)
                    simple_classes = [str(i) for i in range(simple_cm.shape[0])]
                    
                    plot_confusion_matrix(None, None, classes=simple_classes, 
                                         title=f'SNR Index Confusion Matrix (Epoch {epoch})' if epoch else 'SNR Index Confusion Matrix',
                                         normalize=True, 
                                         save_path=os.path.join(save_dir, f'snr_index_conf_matrix_epoch_{epoch}.png') if epoch else None,
                                         cm=simple_cm)
                    return
            except Exception as e:
                print(f"ERROR creating simplified confusion matrix: {str(e)}")
                return
        
        # Map values to indices in the confusion matrix
        for i in range(len(true_snr_values)):
            try:
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
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
        
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
    
    if not use_curriculum or use_curriculum and np.sum(snr_cm) == 0:
        # Use automatic confusion matrix from sklearn
        try:
            # Filter out any NaN or inf values before processing
            valid_mask = np.isfinite(true_snr_values) & np.isfinite(pred_snr_values)
            if np.sum(valid_mask) < len(true_snr_values):
                print(f"WARNING: Filtering out {len(true_snr_values) - np.sum(valid_mask)} non-finite values")
                true_snr_values = true_snr_values[valid_mask]
                pred_snr_values = pred_snr_values[valid_mask]
            
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
                # Calculate maximum index for safety
                max_idx = max(np.max(true_snr_indices), np.max(pred_snr_indices))
                if max_idx < 30:  # Reasonable for safety
                    simple_cm = confusion_matrix(true_snr_indices, pred_snr_indices)
                    simple_classes = [str(i) for i in range(simple_cm.shape[0])]
                    
                    plot_confusion_matrix(None, None, classes=simple_classes, 
                                        title=f'SNR Index Confusion Matrix (Epoch {epoch})' if epoch else 'SNR Index Confusion Matrix',
                                        normalize=True, 
                                        save_path=os.path.join(save_dir, f'snr_index_conf_matrix_epoch_{epoch}.png') if epoch else None,
                                        cm=simple_cm)
                else:
                    print(f"Index range too large ({max_idx}) for safe confusion matrix calculation")
            except Exception as e2:
                print(f"ERROR creating simplified SNR confusion matrix: {str(e2)}")
                print("Unable to create SNR confusion matrix.")
