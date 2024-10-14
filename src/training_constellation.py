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
        total = 0

        # Accumulate true and predicted labels for confusion matrix
        all_true_modulation_labels = []
        all_pred_modulation_labels = []
        all_true_snr_labels = []
        all_pred_snr_labels = []

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False) as progress:
            for inputs, modulation_labels, snr_labels in progress:
                inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)
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

                # Append true and predicted labels for confusion matrix
                all_true_modulation_labels.extend(modulation_labels.cpu().numpy())
                all_pred_modulation_labels.extend(predicted_modulation.cpu().numpy())
                all_true_snr_labels.extend(snr_labels.cpu().numpy())
                all_pred_snr_labels.extend(predicted_snr.cpu().numpy())

                progress.set_postfix(loss=total_loss.item(), mod_accuracy=100.0 * correct_modulation / total, snr_accuracy=100.0 * correct_snr / total)

        train_modulation_accuracy = 100.0 * correct_modulation / total
        train_snr_accuracy = 100.0 * correct_snr / total
        train_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Modulation Accuracy: {train_modulation_accuracy:.2f}%, SNR Accuracy: {train_snr_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        val_loss, val_modulation_accuracy, val_snr_accuracy = validate(model, device, criterion_modulation, criterion_snr, val_loader)

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss)

        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        # Plot confusion matrices for both modulation and SNR
        # Extract label names from dataset
        modulation_labels = list(val_loader.dataset.modulation_labels.keys())  # Extract modulation label names
        snr_labels = list(map(str, val_loader.dataset.snr_labels.keys()))  # Extract SNR labels as strings

        # In the training loop, when calling plot_confusion_matrix:
        plot_confusion_matrix(all_true_modulation_labels, all_pred_modulation_labels, 'Modulation', epoch, label_names=modulation_labels)
        plot_confusion_matrix(all_true_snr_labels, all_pred_snr_labels, 'SNR', epoch, label_names=snr_labels)

        print(f"Validation Loss: {val_loss:.4f}, Modulation Accuracy: {val_modulation_accuracy:.2f}%, SNR Accuracy: {val_snr_accuracy:.2f}%")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_modulation_accuracy": train_modulation_accuracy,
            "train_snr_accuracy": train_snr_accuracy,
            "val_loss": val_loss,
            "val_modulation_accuracy": val_modulation_accuracy,
            "val_snr_accuracy": val_snr_accuracy
        })
