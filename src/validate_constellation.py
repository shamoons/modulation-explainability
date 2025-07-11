# src/validate_constellation.py

import torch
from tqdm import tqdm
from torch.amp import autocast


def validate(model, device, val_loader, criterion_modulation, criterion_snr, uncertainty_weighter=None, use_autocast=False):
    model.eval()
    
    val_loss = 0.0
    modulation_loss_total = 0.0
    snr_loss_total = 0.0
    correct_modulation = 0
    correct_snr = 0
    correct_both = 0
    total = 0

    all_true_modulation_labels = []
    all_pred_modulation_labels = []
    all_true_snr_labels = []
    all_pred_snr_labels = []

    # Get device from model
    device = next(model.parameters()).device
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as progress:
            for inputs, modulation_labels, snr_labels in progress:
                inputs = inputs.to(device)
                modulation_labels = modulation_labels.to(device)
                snr_labels = snr_labels.to(device)

                # Forward pass through the model with optional autocast for CUDA
                if use_autocast and device.type == 'cuda':
                    with autocast('cuda'):
                        modulation_output, snr_output = model(inputs)
                else:
                    modulation_output, snr_output = model(inputs)

                # Compute losses for both modulation and SNR classification
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                # SNR classification with distance penalty
                loss_snr = criterion_snr(snr_output, snr_labels)
                
                # Use uncertainty weighting
                if uncertainty_weighter is not None:
                    total_loss, _ = uncertainty_weighter([loss_modulation, loss_snr])
                else:
                    # Fallback - just sum the losses
                    total_loss = loss_modulation + loss_snr
                
                val_loss += total_loss.item()
                modulation_loss_total += loss_modulation.item()
                snr_loss_total += loss_snr.item()

                # Predict labels
                _, predicted_modulation = modulation_output.max(1)
                
                # Standard classification for SNR - use argmax
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()
                correct_both += ((predicted_modulation == modulation_labels) & (predicted_snr == snr_labels)).sum().item()

                # Collect true and predicted labels for confusion matrix and F1 score computation
                all_true_modulation_labels.extend(modulation_labels.cpu().numpy())
                all_pred_modulation_labels.extend(predicted_modulation.cpu().numpy())
                all_true_snr_labels.extend(snr_labels.cpu().numpy())
                all_pred_snr_labels.extend(predicted_snr.cpu().numpy())

                # Update progress bar with current metrics
                progress.set_postfix(
                    loss=total_loss.item(),
                    mod_accuracy=100.0 * correct_modulation / total,
                    snr_accuracy=100.0 * correct_snr / total,
                    combined_accuracy=100.0 * correct_both / total
                )

    # Compute final validation accuracies and loss
    val_modulation_accuracy = 100.0 * correct_modulation / total
    val_snr_accuracy = 100.0 * correct_snr / total
    val_combined_accuracy = 100.0 * correct_both / total
    val_loss = val_loss / len(val_loader)
    modulation_loss_total /= len(val_loader)
    snr_loss_total /= len(val_loader)

    return (
        val_loss,
        modulation_loss_total,
        snr_loss_total,
        val_modulation_accuracy,
        val_snr_accuracy,
        val_combined_accuracy,
        all_true_modulation_labels,
        all_pred_modulation_labels,
        all_true_snr_labels,
        all_pred_snr_labels
    )