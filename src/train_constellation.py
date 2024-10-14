# src/train_constellation.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.constellation_model import ConstellationResNet
from constellation_loader import ConstellationDataset
from utils.device_utils import get_device
from training_constellation import train
from losses.distance_snr_loss import DistancePenaltyCategoricalSNRLoss  # Import custom SNR loss
import argparse
import warnings
import os

warnings.filterwarnings("ignore", message=r".*NNPACK.*")


def main(checkpoint=None):
    # Load data
    print("Loading data...")

    batch_size = 256
    image_type = 'grayscale'
    root_dir = "constellation"

    # Load full dataset (without splitting)
    dataset = ConstellationDataset(root_dir=root_dir, image_type=image_type)

    # Get train/validation split indices
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Print the number of training and validation samples
    print(f"Number of training samples: {len(train_idx)}")
    print(f"Number of validation samples: {len(val_idx)}")

    # Create samplers for train and validation sets
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Data loaders for training and validation
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12, pin_memory=True)

    # Determine input channels based on image_type
    input_channels = 1 if image_type == 'grayscale' else 3

    # Initialize model with two output heads (modulation and SNR)
    model = ConstellationResNet(num_classes=24, snr_classes=26, input_channels=input_channels)

    # If checkpoint is provided, load the existing model state
    if checkpoint is not None and os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("No checkpoint provided, starting training from scratch.")

    # Initialize loss functions
    criterion_modulation = nn.CrossEntropyLoss()  # Modulation classification loss
    criterion_snr = DistancePenaltyCategoricalSNRLoss()  # Custom SNR loss

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Add learning rate scheduler (Remove verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Determine device (CUDA or CPU)
    device = get_device()

    # Train and validate the model
    epochs = 50
    train(
        model,
        device,
        criterion_modulation,
        criterion_snr,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        epochs=epochs,
        image_type=image_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train constellation model with optional checkpoint loading')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing model checkpoint to resume training', default=None)
    args = parser.parse_args()

    main(checkpoint=args.checkpoint)
