# src/validate_constellation.py

import torch
from tqdm import tqdm
import numpy as np


def validate(model, device, criterion_modulation, criterion_snr, val_loader):
    model.eval()
    val_loss = 0.0
    correct_modulation = 0
    correct_snr = 0
    correct_both = 0
    total = 0

    all_true_modulation_labels = []
    all_pred_modulation_labels = []
    all_true_snr_labels = []
    all_pred_snr_labels = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as progress:
            for inputs, modulation_labels, snr_labels in progress:
                inputs = inputs.to(device)
                modulation_labels = modulation_labels.to(device)
                snr_labels = snr_labels.to(device)

                # Forward pass
                modulation_output, snr_output = model(inputs)

                # Compute loss
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)
                total_loss = loss_modulation + loss_snr
                val_loss += total_loss.item()

                # Predict labels
                _, predicted_modulation = modulation_output.max(1)
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()
                correct_both += ((predicted_modulation == modulation_labels) & (predicted_snr == snr_labels)).sum().item()

                # Convert to numpy arrays
                modulation_labels_np = modulation_labels.cpu().numpy()
                predicted_modulation_np = predicted_modulation.cpu().numpy()
                snr_labels_np = snr_labels.cpu().numpy()
                predicted_snr_np = predicted_snr.cpu().numpy()

                # Collect labels for confusion matrices
                all_true_modulation_labels.extend(modulation_labels_np)
                all_pred_modulation_labels.extend(predicted_modulation_np)
                all_true_snr_labels.extend(snr_labels_np)
                all_pred_snr_labels.extend(predicted_snr_np)

                progress.set_postfix(
                    loss=total_loss.item(),
                    mod_accuracy=100.0 * correct_modulation / total,
                    snr_accuracy=100.0 * correct_snr / total,
                    combined_accuracy=100.0 * correct_both / total
                )

    val_modulation_accuracy = 100.0 * correct_modulation / total
    val_snr_accuracy = 100.0 * correct_snr / total
    val_combined_accuracy = 100.0 * correct_both / total
    val_loss = val_loss / len(val_loader)

    return (
        val_loss,
        val_modulation_accuracy,
        val_snr_accuracy,
        val_combined_accuracy,
        all_true_modulation_labels,
        all_pred_modulation_labels,
        all_true_snr_labels,
        all_pred_snr_labels,
    )
