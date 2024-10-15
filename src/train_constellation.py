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


def main(checkpoint=None, batch_size=64, snr_list=None, epochs=100, warmup_epochs=5):
    # Load data
    print("Loading data...")

    image_type = 'grayscale'
    root_dir = "constellation"

    # Parse snr_list if provided
    if snr_list is not None:
        snr_list = [int(s.strip()) for s in snr_list.split(',')]
    else:
        snr_list = None  # Load all SNRs

    # Load full dataset (without splitting)
    dataset = ConstellationDataset(root_dir=root_dir, image_type=image_type, snr_list=snr_list)

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
    num_modulation_classes = len(dataset.modulation_labels)
    num_snr_classes = len(dataset.snr_labels)
    model = ConstellationResNet(
        num_classes=num_modulation_classes,
        snr_classes=num_snr_classes,
        input_channels=input_channels
    )

    # If checkpoint is provided, load the existing model state
    if checkpoint is not None and os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("No checkpoint provided, starting training from scratch.")

    # Initialize loss functions
    criterion_modulation = nn.CrossEntropyLoss()  # Modulation classification loss
    criterion_snr = nn.CrossEntropyLoss()  # Custom SNR loss

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Calculate the number of batches per epoch
    batches_per_epoch = len(train_loader)

    # Calculate step_size_up as 10% of the total number of epochs
    step_size_up = int(0.1 * epochs * batches_per_epoch)  # 10% of epochs

    # Calculate step_size_down as 20% of the total number of epochs
    step_size_down = int(0.2 * epochs * batches_per_epoch)  # 20% of epochs
    # Add learning rate scheduler with dynamic step_size_up
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular2', cycle_momentum=False)

    # Determine device (CUDA, MPS, or CPU)
    device = get_device()

    # Train and validate the model
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
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation', default=64)
    parser.add_argument('--snr_list', type=str, help='Comma-separated list of SNR values to load', default=None)
    parser.add_argument('--epochs', type=int, help='Total number of epochs for training', default=100)
    parser.add_argument('--warmup_epochs', type=int, help='Number of warm-up epochs for learning rate', default=5)
    args = parser.parse_args()

    main(checkpoint=args.checkpoint, batch_size=args.batch_size, snr_list=args.snr_list, epochs=args.epochs, warmup_epochs=args.warmup_epochs)
