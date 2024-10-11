# src/training_constellation.py
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def train(model, device, criterion, optimizer, train_loader, val_loader, epochs=10):
    """
    Train the model and validate it at the end of each epoch.
    Includes gradient clipping and logs metrics with Weights and Biases (wandb).

    Args:
        model: The PyTorch model.
        device: The device to run the model on (e.g., "cuda" or "cpu").
        criterion: The loss function.
        optimizer: The optimizer for backpropagation.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        epochs: Number of epochs to train.
    """
    # Initialize WandB project
    wandb.init(project="modulation-explainability", config={"epochs": epochs})

    # Extract labels from the dataset in val_loader (assuming labels are stored in the dataset)
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

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar with current loss
                progress.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # Log to Weights and Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        val_loss, val_accuracy, val_predictions, val_targets = validate(model, device, criterion, val_loader)

        # Generate confusion matrix and save it as an image with labels
        save_confusion_matrix(val_targets, val_predictions, epoch, labels=label_names)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


def validate(model, device, criterion, val_loader):
    """
    Validate the model on the validation dataset with tqdm progress bar.

    Args:
        model: The PyTorch model.
        device: The device to run the model on (e.g., "cuda" or "cpu").
        criterion: The loss function.
        val_loader: DataLoader for the validation data.

    Returns:
        val_loss: The validation loss.
        val_accuracy: The validation accuracy.
        val_predictions: List of predicted labels for confusion matrix.
        val_targets: List of true labels for confusion matrix.
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

                # Store predictions and true labels for confusion matrix
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

    Args:
        targets: List of true labels.
        predictions: List of predicted labels.
        epoch: Current epoch number (used in the filename).
        labels: Optional list of labels for the confusion matrix. If None, uses numeric labels.
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
