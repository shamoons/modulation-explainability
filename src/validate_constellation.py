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
