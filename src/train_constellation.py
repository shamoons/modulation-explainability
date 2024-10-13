# src/train_constellation.py
import torch.nn as nn
import torch.optim as optim
from models.constellation_model import ConstellationCNN  # Import ConstellationCNN model
from constellation_loader import get_constellation_dataloader  # Import function for loading constellation images
from utils import get_device  # Function to get the device (CPU or GPU)
from training_constellation import train  # Import the training and validation function
import warnings
warnings.filterwarnings("ignore", message=r".*NNPACK.*")


if __name__ == "__main__":
    # Load data
    print("Loading data...")

    # Get the DataLoaders for training, validation, and testing
    batch_size = 32
    input_size = (224, 224)  # Constellation image size

    # Load train, validation, and test sets
    train_loader = get_constellation_dataloader(root_dir="constellation", batch_size=batch_size)
    val_loader = get_constellation_dataloader(root_dir="constellation", batch_size=batch_size, shuffle=False)

    # Print the number of samples in each set
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")

    # Initialize model, loss function, and optimizer
    model = ConstellationCNN(num_classes=24, input_size=input_size)  # 24 modulation classes for this example
    criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Determine device (CUDA or CPU)
    device = get_device()

    # Train and validate the model
    epochs = 50
    train(model, device, criterion, optimizer, train_loader, val_loader, epochs=epochs)
