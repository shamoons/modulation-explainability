# src/train_constellation.py
import torch.nn as nn
import torch.optim as optim
from models.constellation_model import ConstellationResNet  # Import the updated model
from constellation_loader import get_constellation_dataloader  # Import function for loading constellation images
from utils import get_device  # Function to get the device (CPU or GPU)
from training_constellation import train  # Import the training and validation function
import warnings

warnings.filterwarnings("ignore", message=r".*NNPACK.*")


if __name__ == "__main__":
    # Load data
    print("Loading data...")

    # Set parameters
    batch_size = 32
    image_type = 'grayscale'  # Choose 'three_channel' or 'grayscale'

    # Determine input_channels based on image_type
    if image_type == 'grayscale':
        input_channels = 1
    elif image_type == 'three_channel':
        input_channels = 3
    else:
        raise ValueError(f"Unsupported image_type '{image_type}'. Supported types are 'three_channel' and 'grayscale'.")

    # Get the DataLoaders for training and validation
    train_loader = get_constellation_dataloader(
        root_dir="constellation",
        batch_size=batch_size,
        image_type=image_type
    )
    val_loader = get_constellation_dataloader(
        root_dir="constellation",
        batch_size=batch_size,
        shuffle=False,
        image_type=image_type
    )

    # Print the number of samples in each set
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")

    # Initialize model, loss function, and optimizer
    model = ConstellationResNet(
        num_classes=24,  # Adjust based on the number of modulation classes
        input_channels=input_channels,  # Use input_channels determined by image_type
        pretrained=False  # Set to True if you want to use pretrained weights
    )
    criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Determine device (CUDA or CPU)
    device = get_device()

    # Train and validate the model
    epochs = 50
    train(
        model,
        device,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        epochs=epochs,
        image_type=image_type  # Pass image_type to the training function
    )
