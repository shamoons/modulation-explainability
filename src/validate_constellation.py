# src/validate_constellation.py

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast
from utils.config_utils import load_loss_config


# src/validate_constellation.py

def validate(model, device, criterion_modulation, criterion_snr, criterion_dynamic, val_loader, use_autocast=False):
    """
    Validate the model on the validation set.
    Returns validation loss, accuracies, and predictions for plotting.
    """
    model.eval()
    val_loss = 0.0
    modulation_loss_total = 0.0
    snr_loss_total = 0.0
    correct_modulation = 0
    snr_mae = 0.0  # Mean Absolute Error for SNR
    total = 0

    # Lists to store predictions and true labels for plotting
    all_pred_modulation_labels = []
    all_true_modulation_labels = []
    all_pred_snr_labels = []
    all_true_snr_labels = []

    with torch.no_grad():
        for inputs, modulation_labels, snr_labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            modulation_labels = modulation_labels.to(device)
            snr_labels = snr_labels.to(device)

            if use_autocast:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    modulation_output, snr_output = model(inputs)
                    loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                    loss_snr = criterion_snr(snr_output, snr_labels)
                    loss = criterion_dynamic([loss_modulation, loss_snr])
            else:
                modulation_output, snr_output = model(inputs)
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)
                loss = criterion_dynamic([loss_modulation, loss_snr])

            val_loss += loss.item()
            modulation_loss_total += loss_modulation.item()
            snr_loss_total += loss_snr.item()

            _, predicted_modulation = modulation_output.max(1)
            total += modulation_labels.size(0)
            correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()

            # Calculate SNR MAE using the expected value from probabilities
            probs = torch.softmax(snr_output, dim=1)
            expected_snr = torch.sum(probs * criterion_snr.snr_values.to(device), dim=1)
            snr_mae += torch.abs(expected_snr - criterion_snr.snr_values[snr_labels].to(device)).mean().item()

            # Store predictions and true labels
            all_pred_modulation_labels.extend(predicted_modulation.cpu().numpy())
            all_true_modulation_labels.extend(modulation_labels.cpu().numpy())
            all_pred_snr_labels.extend(expected_snr.cpu().numpy())
            all_true_snr_labels.extend(criterion_snr.snr_values[snr_labels].cpu().numpy())

    # Calculate average loss and accuracy
    val_loss = val_loss / len(val_loader)
    modulation_loss_total = modulation_loss_total / len(val_loader)
    snr_loss_total = snr_loss_total / len(val_loader)
    val_modulation_accuracy = 100.0 * correct_modulation / total
    val_snr_mae = snr_mae / len(val_loader)

    return (
        val_loss,
        modulation_loss_total,
        snr_loss_total,
        val_modulation_accuracy,
        val_snr_mae,
        all_true_modulation_labels,
        all_pred_modulation_labels,
        all_true_snr_labels,
        all_pred_snr_labels
    )

def plot_validation_confusion_matrices(true_mod_labels, pred_mod_labels, true_snr_values, pred_snr_values, 
                                      mod_classes=None, save_dir=None, epoch=None):
    """
    Plot and optionally save confusion matrices for modulation classification and SNR prediction.
    
    Args:
        true_mod_labels: True modulation class indices
        pred_mod_labels: Predicted modulation class indices
        true_snr_values: True SNR values in dB
        pred_snr_values: Predicted SNR values in dB
        mod_classes: List of modulation class names (optional)
        save_dir: Directory to save plots (if None, just display)
        epoch: Current epoch for naming saved files
    """
    # Convert inputs to numpy arrays if they're not already
    true_mod_labels = np.array(true_mod_labels)
    pred_mod_labels = np.array(pred_mod_labels)
    true_snr_values = np.array(true_snr_values)
    pred_snr_values = np.squeeze(np.array(pred_snr_values))  # Ensure it's 1D
    
    # 1. Modulation Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_mod = confusion_matrix(true_mod_labels, pred_mod_labels)
    
    # Normalize by row (true labels) for better visualization
    cm_mod_norm = cm_mod.astype('float') / cm_mod.sum(axis=1)[:, np.newaxis]
    cm_mod_norm = np.nan_to_num(cm_mod_norm)  # Replace NaNs with zeros
    
    # Plot with class names if provided, otherwise use indices
    if mod_classes:
        sns.heatmap(cm_mod_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=mod_classes, yticklabels=mod_classes)
    else:
        sns.heatmap(cm_mod_norm, annot=True, fmt='.2f', cmap='Blues')
    
    plt.title('Modulation Classification Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir and epoch is not None:
        plt.savefig(f"{save_dir}/modulation_cm_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()
    
    # 2. SNR Confusion Matrix
    
    # Round SNR values to nearest dB for binning
    true_snr_rounded = np.round(true_snr_values)
    pred_snr_rounded = np.round(pred_snr_values)
    
    # Get unique sorted SNR values (using set union)
    all_snr_values = np.sort(np.unique(np.concatenate([true_snr_rounded, pred_snr_rounded])))
    
    # Create mapping from SNR values to indices
    snr_to_idx = {snr: i for i, snr in enumerate(all_snr_values)}
    
    # Map to indices for confusion matrix
    true_snr_idx = np.array([snr_to_idx[snr] for snr in true_snr_rounded])
    pred_snr_idx = np.array([snr_to_idx[snr] for snr in pred_snr_rounded])
    
    # Generate confusion matrix
    plt.figure(figsize=(12, 10))
    cm_snr = confusion_matrix(true_snr_idx, pred_snr_idx)
    
    # Normalize by row
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_snr_norm = cm_snr.astype('float') / cm_snr.sum(axis=1)[:, np.newaxis]
        cm_snr_norm = np.nan_to_num(cm_snr_norm)  # Replace NaNs with zeros
    
    # Plot
    sns.heatmap(cm_snr_norm, annot=False, cmap='Blues',
               xticklabels=all_snr_values, yticklabels=all_snr_values)
    
    plt.title('SNR Prediction Confusion Matrix')
    plt.xlabel('Predicted SNR (dB)')
    plt.ylabel('True SNR (dB)')
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir and epoch is not None:
        plt.savefig(f"{save_dir}/snr_cm_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()
        
    # 3. Distribution of SNR prediction errors
    plt.figure(figsize=(10, 6))
    
    # Calculate errors
    errors = pred_snr_values - true_snr_values
    
    # Plot histogram of errors
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--')
    
    # Add vertical lines at +/- 2dB
    plt.axvline(x=2, color='g', linestyle='--')
    plt.axvline(x=-2, color='g', linestyle='--')
    
    # Calculate percentage of predictions within +/- 2dB
    within_2db = np.sum(np.abs(errors) <= 2.0) / len(errors) * 100
    
    plt.title(f'SNR Prediction Error Distribution - {within_2db:.2f}% within ±2dB')
    plt.xlabel('Prediction Error (dB)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir and epoch is not None:
        plt.savefig(f"{save_dir}/snr_error_dist_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()
