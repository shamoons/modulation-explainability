# src/training_constellation.py

import torch
import wandb
from utils.image_utils import plot_confusion_matrix
from validate_constellation import validate
from tqdm import tqdm
import os


def train(model, device, criterion_modulation, criterion_snr, optimizer, scheduler, train_loader, val_loader, epochs=10, image_type='three_channel', save_dir="checkpoints"):
    """
    Train the model and save the best one based on validation loss.
    Log F1 score after validation step and plot F1 score for each class.
    """
    # Initialize WandB project and log image_type
    wandb.init(project="modulation-explainability", config={"epochs": epochs, "image_type": image_type})

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Initialize best validation loss

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_modulation = 0
        correct_snr = 0
        correct_both = 0  # For combined accuracy
        total = 0

        # Accumulate true and predicted labels for confusion matrix
        all_true_modulation_labels = []
        all_pred_modulation_labels = []
        all_true_snr_labels = []
        all_pred_snr_labels = []

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False) as progress:
            for inputs, modulation_labels, snr_labels in progress:
                inputs = inputs.to(device)
                modulation_labels = modulation_labels.to(device)
                snr_labels = snr_labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                modulation_output, snr_output = model(inputs)

                # Compute loss for both outputs
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)

                # Combine both losses
                total_loss = loss_modulation + loss_snr
                total_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += total_loss.item()

                _, predicted_modulation = modulation_output.max(1)
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()

                # Calculate combined accuracy
                correct_both += ((predicted_modulation == modulation_labels) & (predicted_snr == snr_labels)).sum().item()

                # Append true and predicted labels for confusion matrix
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

        train_modulation_accuracy = 100.0 * correct_modulation / total
        train_snr_accuracy = 100.0 * correct_snr / total
        train_combined_accuracy = 100.0 * correct_both / total
        train_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Modulation Accuracy: {train_modulation_accuracy:.2f}%, SNR Accuracy: {train_snr_accuracy:.2f}%, Combined Accuracy: {train_combined_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        val_results = validate(model, device, criterion_modulation, criterion_snr, val_loader)
        val_loss, val_modulation_accuracy, val_snr_accuracy, val_combined_accuracy, all_true_combined_labels, all_pred_combined_labels, actual_combinations = val_results

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss)

        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        # Map indices back to label names using inverse mappings from the dataset
        modulation_label_names = [train_loader.dataset.inverse_modulation_labels[idx] for idx in range(len(train_loader.dataset.modulation_labels))]
        snr_label_names = [str(train_loader.dataset.inverse_snr_labels[idx]) for idx in range(len(train_loader.dataset.snr_labels))]

        # Plot confusion matrices with only the labels present in the data
        plot_confusion_matrix(
            all_true_modulation_labels,
            all_pred_modulation_labels,
            'Modulation',
            epoch,
            label_names=modulation_label_names
        )
        plot_confusion_matrix(
            all_true_snr_labels,
            all_pred_snr_labels,
            'SNR',
            epoch,
            label_names=snr_label_names
        )

        # Generate combined modulation and SNR labels based on actual combinations
        mod_labels = []
        snr_labels = []
        combination_to_index = {}
        index = 0
        N_snr = len(snr_label_names)

        for mod_idx, mod_name in enumerate(modulation_label_names):
            for snr_idx, snr_value in enumerate(snr_label_names):
                if (mod_idx, snr_idx) in actual_combinations:
                    mod_labels.append(mod_name)
                    snr_labels.append(snr_value)
                    combination_to_index[(mod_idx, snr_idx)] = index
                    index += 1

        # Map combined labels to new indices
        mapped_true_combined_labels = []
        mapped_pred_combined_labels = []
        for tru_mod in all_true_combined_labels:
            mod_idx = int(tru_mod // N_snr)
            snr_idx = int(tru_mod % N_snr)
            if (mod_idx, snr_idx) in combination_to_index:
                mapped_true_combined_labels.append(combination_to_index[(mod_idx, snr_idx)])
        for pred_mod in all_pred_combined_labels:
            mod_idx = int(pred_mod // N_snr)
            snr_idx = int(pred_mod % N_snr)
            if (mod_idx, snr_idx) in combination_to_index:
                mapped_pred_combined_labels.append(combination_to_index[(mod_idx, snr_idx)])

        # Plot combined confusion matrix with multi-level labels
        plot_confusion_matrix(
            mapped_true_combined_labels,
            mapped_pred_combined_labels,
            'Combined',
            epoch,
            label_names=None,  # Not used in this case
            mod_labels=mod_labels,
            snr_labels=snr_labels
        )

        print(f"Validation Loss: {val_loss:.4f}, Modulation Accuracy: {val_modulation_accuracy:.2f}%, SNR Accuracy: {val_snr_accuracy:.2f}%, Combined Accuracy: {val_combined_accuracy:.2f}%")

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_modulation_accuracy": train_modulation_accuracy,
            "train_snr_accuracy": train_snr_accuracy,
            "train_combined_accuracy": train_combined_accuracy,
            "val_loss": val_loss,
            "val_modulation_accuracy": val_modulation_accuracy,
            "val_snr_accuracy": val_snr_accuracy,
            "val_combined_accuracy": val_combined_accuracy
        })
