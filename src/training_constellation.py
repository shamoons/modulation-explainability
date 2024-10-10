# src/training_constellation.py
import torch
import wandb


def train(model, device, criterion, optimizer, train_loader, val_loader, epochs=10, scheduler=None, patience=5):
    """
    Train the model and validate it at the end of each epoch.
    Includes early stopping, learning rate scheduling, and gradient clipping.
    Logs metrics with Weights and Biases (wandb).

    Args:
        model: The PyTorch model.
        device: The device to run the model on (e.g., "cuda" or "cpu").
        criterion: The loss function.
        optimizer: The optimizer for backpropagation.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        epochs: Number of epochs to train.
        scheduler: Optional learning rate scheduler.
        patience: Patience for early stopping.
    """
    wandb.init(project="modulation-explainability", config={"epochs": epochs})
    model.to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_accuracy = validate(model, device, criterion, val_loader)

        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the model if it has improved
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved model with validation loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{patience}")

        # Stop training early if no improvement after 'patience' epochs
        if early_stopping_counter >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

        # Log to Weights and Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']  # Log the learning rate
        })

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


def validate(model, device, criterion, val_loader):
    """
    Validate the model on the validation dataset.

    Args:
        model: The PyTorch model.
        device: The device to run the model on (e.g., "cuda" or "cpu").
        criterion: The loss function.
        val_loader: DataLoader for the validation data.

    Returns:
        val_loss: The validation loss.
        val_accuracy: The validation accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100.0 * correct / total
    val_loss = val_loss / len(val_loader)
    return val_loss, val_accuracy
