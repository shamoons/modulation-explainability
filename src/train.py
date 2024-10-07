# src/train.py
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import LightweightCNN
from data_loader import get_dataloaders  # Updated function for batched loading
from utils import get_device
from training import train, validate  # Add validate function for evaluation

if __name__ == "__main__":
    # Load data
    print("Loading data...")

    # Get the DataLoaders for training, validation, and testing
    batch_size = 64
    train_loader, val_loader, test_loader, mod2int = get_dataloaders(batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = LightweightCNN(num_classes=24)  # Updated for 24 modulation classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Determine device (CUDA, MPS, or CPU)
    device = get_device()

    # Train the model
    epochs = 50
    train(model, device, criterion, optimizer, train_loader, val_loader, epochs=epochs)

    # Evaluate the model on validation set after training
    val_loss, val_accuracy = validate(model, device, criterion, val_loader)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Optionally: Add evaluation on the test set
    test_loss, test_accuracy = validate(model, device, criterion, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
