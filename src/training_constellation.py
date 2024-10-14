# src/training_constellation.py
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import os


def train(model, device, criterion_modulation, criterion_snr, optimizer, train_loader, val_loader, epochs=10, image_type='three_channel', save_dir="checkpoints"):
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

        modulation_predictions = []
        modulation_targets = []
        snr_predictions = []
        snr_targets = []

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False) as progress:
            for inputs, (modulation_labels, snr_labels) in progress:
                inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                modulation_output, snr_output = model(inputs)

                # Compute loss for both outputs
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)

                # Total loss is a combination of both losses
                loss = loss_modulation + loss_snr
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                _, predicted_modulation = modulation_output.max(1)
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()

                # Track predictions and true labels for confusion matrix and F1
                modulation_predictions.extend(predicted_modulation.cpu().numpy())
                modulation_targets.extend(modulation_labels.cpu().numpy())
                snr_predictions.extend(predicted_snr.cpu().numpy())
                snr_targets.extend(snr_labels.cpu().numpy())

                progress.set_postfix(loss=loss.item(), mod_accuracy=100.0 * correct_modulation / total, snr_accuracy=100.0 * correct_snr / total)

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

        # Compute and log confusion matrix and F1 scores
        log_confusion_matrix(modulation_targets, modulation_predictions, "Modulation", epoch)
        log_confusion_matrix(snr_targets, snr_predictions, "SNR", epoch)

        modulation_f1_scores = precision_recall_fscore_support(modulation_targets, modulation_predictions, average=None)[2]
        snr_f1_scores = precision_recall_fscore_support(snr_targets, snr_predictions, average=None)[2]

        plot_f1_scores(modulation_f1_scores, label_type="Modulation", epoch=epoch)
        plot_f1_scores(snr_f1_scores, label_type="SNR", epoch=epoch)

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

    modulation_predictions = []
    modulation_targets = []
    snr_predictions = []
    snr_targets = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as progress:
            for inputs, (modulation_labels, snr_labels) in progress:
                inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)

                # Forward pass
                modulation_output, snr_output = model(inputs)

                # Compute loss for both outputs
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)

                # Total loss
                loss = loss_modulation + loss_snr
                val_loss += loss.item()

                _, predicted_modulation = modulation_output.max(1)
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()

                # Track predictions and true labels for confusion matrix and F1
                modulation_predictions.extend(predicted_modulation.cpu().numpy())
                modulation_targets.extend(modulation_labels.cpu().numpy())
                snr_predictions.extend(predicted_snr.cpu().numpy())
                snr_targets.extend(snr_labels.cpu().numpy())

                progress.set_postfix(loss=loss.item(), mod_accuracy=100.0 * correct_modulation / total, snr_accuracy=100.0 * correct_snr / total)

    val_modulation_accuracy = 100.0 * correct_modulation / total
    val_snr_accuracy = 100.0 * correct_snr / total
    val_loss = val_loss / len(val_loader)

    # Compute and log confusion matrix and F1 scores for validation
    log_confusion_matrix(modulation_targets, modulation_predictions, "Modulation", "Validation")
    log_confusion_matrix(snr_targets, snr_predictions, "SNR", "Validation")

    modulation_f1_scores = precision_recall_fscore_support(modulation_targets, modulation_predictions, average=None)[2]
    snr_f1_scores = precision_recall_fscore_support(snr_targets, snr_predictions, average=None)[2]

    plot_f1_scores(modulation_f1_scores, label_type="Modulation", epoch="Validation")
    plot_f1_scores(snr_f1_scores, label_type="SNR", epoch="Validation")

    return val_loss, val_modulation_accuracy, val_snr_accuracy


def log_confusion_matrix(targets, predictions, label_type, epoch):
    """
    Log confusion matrix as an image using matplotlib and save it.
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{label_type} Confusion Matrix - Epoch {epoch}")

    # Save confusion matrix
    plt.savefig(f"confusion_matrix_{label_type}_epoch_{epoch}.png")
    plt.close()

    # Log to WandB
    wandb.log({f"{label_type} Confusion Matrix Epoch {epoch}": wandb.Image(f"confusion_matrix_{label_type}_epoch_{epoch}.png")})


def plot_f1_scores(f1_scores, label_type, epoch):
    """
    Plot F1 scores for each class and save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(f1_scores)), f1_scores, color='b', alpha=0.7)
    plt.xlabel(f'{label_type} Classes')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores for {label_type} Classes - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'f1_scores_{label_type}_epoch_{epoch}.png')
    plt.close()

    # Log to WandB
    wandb.log({f"F1 Scores {label_type} Epoch {epoch}": wandb.Image(f'f1_scores_{label_type}_epoch_{epoch}.png')})
