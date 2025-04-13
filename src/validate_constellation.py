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
    correct_snr = 0
    total = 0

    # Lists to store predictions and true labels for plotting
    all_pred_modulation_labels = []
    all_true_modulation_labels = []
    all_pred_snr_indices = []  # Store indices instead of values
    all_true_snr_indices = []  # Store indices instead of values

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

            # Get predicted classes
            _, predicted_modulation = modulation_output.max(1)
            _, predicted_snr = snr_output.max(1)
            
            total += modulation_labels.size(0)
            correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
            correct_snr += predicted_snr.eq(snr_labels).sum().item()

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

    return (
        val_loss,
        modulation_loss_total,
        snr_loss_total,
        val_modulation_accuracy,
        val_snr_accuracy,
        all_true_modulation_labels,
        all_pred_modulation_labels,
        all_true_snr_indices,  # Return indices instead of values
        all_pred_snr_indices   # Return indices instead of values
    )

def plot_validation_confusion_matrices(true_mod_labels, pred_mod_labels, true_snr_indices, pred_snr_indices, 
                                    mod_classes=None, save_dir=None, epoch=None):
    """
    Plot and optionally save confusion matrices for modulation classification and SNR prediction.
    
    Args:
        true_mod_labels: True modulation class indices
        pred_mod_labels: Predicted modulation class indices
        true_snr_indices: True SNR class indices
        pred_snr_indices: Predicted SNR class indices
        mod_classes: List of modulation class names (optional)
        save_dir: Directory to save plots (if None, just display)
        epoch: Current epoch number
    """
    # Convert inputs to numpy arrays if they're not already
    true_mod_labels = np.array(true_mod_labels)
    pred_mod_labels = np.array(pred_mod_labels)
    true_snr_indices = np.array(true_snr_indices)
    pred_snr_indices = np.array(pred_snr_indices)
    
    # 1. Modulation Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_mod = confusion_matrix(true_mod_labels, pred_mod_labels)
    
    # Normalize by row (true labels)
    cm_mod_norm = cm_mod.astype('float') / cm_mod.sum(axis=1)[:, np.newaxis]
    cm_mod_norm = np.nan_to_num(cm_mod_norm)
    
    if mod_classes:
        sns.heatmap(cm_mod_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=mod_classes, yticklabels=mod_classes)
    else:
        sns.heatmap(cm_mod_norm, annot=True, fmt='.2f', cmap='Blues')
    
    plt.title('Modulation Classification Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    if save_dir and epoch is not None:
        plt.savefig(f"{save_dir}/modulation_cm_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()
    
    # 2. SNR Confusion Matrix
    plt.figure(figsize=(12, 10))
    
    # Create confusion matrix using indices
    cm_snr = confusion_matrix(true_snr_indices, pred_snr_indices)
    
    # Normalize by row
    cm_snr_norm = cm_snr.astype('float') / cm_snr.sum(axis=1)[:, np.newaxis]
    cm_snr_norm = np.nan_to_num(cm_snr_norm)
    
    # Get SNR values for labels
    snr_values = [-20, 0, 30]  # The fixed SNR values we're using
    
    # Plot SNR confusion matrix
    sns.heatmap(cm_snr_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=snr_values, yticklabels=snr_values)
    
    plt.title('SNR Classification Confusion Matrix')
    plt.xlabel('Predicted SNR (dB)')
    plt.ylabel('True SNR (dB)')
    plt.tight_layout()
    
    if save_dir and epoch is not None:
        plt.savefig(f"{save_dir}/snr_cm_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()
    
    # 3. SNR Classification Accuracy Distribution
    plt.figure(figsize=(10, 6))
    
    # Calculate accuracy per SNR class
    snr_accuracies = np.diag(cm_snr_norm) * 100
    
    # Create bar plot of accuracies
    plt.bar(snr_values, snr_accuracies)
    plt.axhline(y=np.mean(snr_accuracies), color='r', linestyle='--', 
                label=f'Mean Accuracy: {np.mean(snr_accuracies):.1f}%')
    
    plt.title('SNR Classification Accuracy by Class')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir and epoch is not None:
        plt.savefig(f"{save_dir}/snr_accuracy_dist_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()
