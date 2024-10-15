# src/validate_constellation.py

import torch
from tqdm import tqdm


def validate(model, device, criterion_modulation, criterion_snr, val_loader):
    """
    Validate the model on the validation dataset with tqdm progress bar.
    """
    model.eval()
    val_loss = 0.0
    correct_modulation = 0
    correct_snr = 0
    correct_both = 0  # For combined accuracy
    total = 0

    all_true_modulation_labels = []
    all_pred_modulation_labels = []
    all_true_snr_labels = []
    all_pred_snr_labels = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as progress:
            for inputs, modulation_labels, snr_labels in progress:
                inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)

                # Forward pass
                modulation_output, snr_output = model(inputs)

                # Compute loss for both outputs
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)

                # Combine both losses
                total_loss = loss_modulation + loss_snr
                val_loss += total_loss.item()

                _, predicted_modulation = modulation_output.max(1)
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()

                # Calculate combined accuracy
                correct_both += ((predicted_modulation == modulation_labels) & (predicted_snr == snr_labels)).sum().item()

                all_true_modulation_labels.extend(modulation_labels.cpu().numpy())
                all_pred_modulation_labels.extend(predicted_modulation.cpu().numpy())
                all_true_snr_labels.extend(snr_labels.cpu().numpy())
                all_pred_snr_labels.extend(predicted_snr.cpu().numpy())

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

    return val_loss, val_modulation_accuracy, val_snr_accuracy, val_combined_accuracy
