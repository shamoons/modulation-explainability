# src/training_constellation.py
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import os


def train(model, device, criterion, optimizer, train_loader, val_loader, epochs=10, image_type='three_channel', save_dir="checkpoints"):
    """
    Train the model and save the best one based on validation loss.
    Log F1 score after validation step and plot F1 score for each class.
    """
    # Initialize WandB project and log image_type
    wandb.init(project="modulation-explainability", config={"epochs": epochs, "image_type": image_type})

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Initialize best validation loss

    label_names = val_loader.dataset.mods_to_process if val_loader.dataset.mods_to_process else sorted(val_loader.dataset.modulation_labels.keys())

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False) as progress:
            for inputs, labels in progress:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        val_loss, val_accuracy, val_predictions, val_targets = validate(model, device, criterion, val_loader)

        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        save_confusion_matrix(val_targets, val_predictions, epoch, labels=label_names)

        # Calculate F1 scores and plot them
        precision, recall, f1_scores, _ = precision_recall_fscore_support(val_targets, val_predictions, average=None)
        plot_f1_scores(f1_scores, label_names, epoch)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


def validate(model, device, criterion, val_loader):
    """
    Validate the model on the validation dataset with tqdm progress bar.
    Also returns precision, recall, and F1 scores for analysis.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    val_predictions = []
    val_targets = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as progress:
            for inputs, labels in progress:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                # Update progress bar with current validation loss and accuracy
                progress.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    val_accuracy = 100.0 * correct / total
    val_loss = val_loss / len(val_loader)

    return val_loss, val_accuracy, val_predictions, val_targets


def save_confusion_matrix(targets, predictions, epoch, labels=None):
    """
    Save the confusion matrix as an image using matplotlib.
    """
    cm = confusion_matrix(targets, predictions)

    # If no labels are provided, use numeric labels
    if labels is None:
        num_classes = cm.shape[0]
        labels = list(range(num_classes))

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Epoch {epoch + 1}")

    # Save confusion matrix
    plt.savefig(f"confusion_matrix_epoch_{epoch + 1}.png")
    plt.close()

    # Log to Weights and Biases
    wandb.log({f"Confusion Matrix Epoch {epoch + 1}": wandb.Image(f"confusion_matrix_epoch_{epoch + 1}.png")})


def plot_f1_scores(f1_scores, labels, epoch):
    """
    Plot F1 scores for each class and save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(labels, f1_scores, color='b', alpha=0.7)
    plt.xticks(rotation=45)
    plt.xlabel('Modulation Classes')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score for Each Modulation Class - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'f1_scores_epoch_{epoch}.png')
    plt.close()

    # Log to Weights and Biases
    wandb.log({f"F1 Scores Epoch {epoch}": wandb.Image(f"f1_scores_epoch_{epoch}.png")})
