import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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

        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        # Plot confusion matrices for both modulation and SNR
        plot_confusion_matrix(all_true_modulation_labels, all_pred_modulation_labels, 'Modulation', epoch)
        plot_confusion_matrix(all_true_snr_labels, all_pred_snr_labels, 'SNR', epoch)

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


def validate(model, device, criterion_modulation, criterion_snr, val_loader):
    """
    Validate the model on the validation dataset with tqdm progress bar.
    """
    model.eval()
    val_loss = 0.0
    correct_modulation = 0
    correct_snr = 0
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

                all_true_modulation_labels.extend(modulation_labels.cpu().numpy())
                all_pred_modulation_labels.extend(predicted_modulation.cpu().numpy())
                all_true_snr_labels.extend(snr_labels.cpu().numpy())
                all_pred_snr_labels.extend(predicted_snr.cpu().numpy())

                progress.set_postfix(loss=total_loss.item(), mod_accuracy=100.0 * correct_modulation / total, snr_accuracy=100.0 * correct_snr / total)

    val_modulation_accuracy = 100.0 * correct_modulation / total
    val_snr_accuracy = 100.0 * correct_snr / total
    val_loss = val_loss / len(val_loader)

    return val_loss, val_modulation_accuracy, val_snr_accuracy


def plot_confusion_matrix(true_labels, pred_labels, label_type, epoch):
    """
    Plot and save a confusion matrix.

    Args:
        true_labels (list of int): True class labels.
        pred_labels (list of int): Predicted class labels.
        label_type (str): Type of label ('Modulation' or 'SNR').
        epoch (int): Current epoch number.
    """
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel(f"Predicted {label_type} Label")
    plt.ylabel(f"True {label_type} Label")
    plt.title(f"{label_type} Confusion Matrix - Epoch {epoch + 1}")

    # Save confusion matrix
    plt.savefig(f"confusion_matrix_{label_type}_epoch_{epoch + 1}.png")
    plt.close()

    # Log to Weights and Biases
    wandb.log({f"Confusion Matrix {label_type} Epoch {epoch + 1}": wandb.Image(f"confusion_matrix_{label_type}_epoch_{epoch + 1}.png")})
