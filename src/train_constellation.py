# src/train_constellation.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.constellation_model import ConstellationResNet
from models.vision_transformer_model import ConstellationVisionTransformer
from constellation_loader import ConstellationDataset
from utils.device_utils import get_device
from training_constellation import train
import argparse
import warnings
import os

warnings.filterwarnings("ignore", message=r".*NNPACK.*")


def main(checkpoint=None, batch_size=64, snr_list=None, mods_to_process=None, epochs=100, use_snr_buckets=False):
    # Load data
    print("Loading data...")

    image_type = 'grayscale'
    root_dir = "constellation"

    # Parse snr_list and mods_to_process if provided
    if snr_list is not None:
        snr_list = [int(s.strip()) for s in snr_list.split(',')]
    else:
        snr_list = None  # Load all SNRs

    if mods_to_process is not None:
        mods_to_process = [mod.strip() for mod in mods_to_process.split(',')]
    else:
        mods_to_process = None  # Load all modulation types

    # Load full dataset (with modulation types and SNRs filtering)
    dataset = ConstellationDataset(root_dir=root_dir, image_type=image_type, snr_list=snr_list, mods_to_process=mods_to_process, use_snr_buckets=use_snr_buckets)

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

    num_snr_classes = len(dataset.snr_labels)
    num_modulation_classes = len(dataset.modulation_labels)

    print(f"Number of modulation classes: {num_modulation_classes}")
    print(f"Number of SNR classes: {num_snr_classes}")

    # model = ConstellationVisionTransformer(
    #     num_classes=num_modulation_classes,
    #     snr_classes=num_snr_classes,
    #     input_channels=input_channels
    # )
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
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # Calculate the number of batches per epoch
    batches_per_epoch = len(train_loader)

    # Define number of cycles
    num_cycles = 8

    # Total number of batches over all epochs
    total_batches = batches_per_epoch * epochs

    # Step size for each phase (up and down) in each cycle
    step_size_up_down = total_batches // (num_cycles * 2)

    # Add learning rate scheduler with dynamic step_size_up
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.0001, step_size_up=step_size_up_down, step_size_down=step_size_up_down, mode='triangular2', cycle_momentum=False)

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
        mod_list=mods_to_process,
        snr_list=snr_list,
        use_snr_buckets=use_snr_buckets
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train constellation model with optional checkpoint loading')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing model checkpoint to resume training', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation', default=64)
    parser.add_argument('--snr_list', type=str, help='Comma-separated list of SNR values to load', default=None)
    parser.add_argument('--mods_to_process', type=str, help='Comma-separated list of modulation types to load', default=None)
    parser.add_argument('--epochs', type=int, help='Total number of epochs for training', default=100)
    parser.add_argument('--use_snr_buckets', action='store_true', help='Flag to use SNR buckets instead of actual SNR values')
    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        epochs=args.epochs,
        use_snr_buckets=args.use_snr_buckets
    )
