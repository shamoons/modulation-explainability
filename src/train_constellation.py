# src/train_constellation.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.constellation_model import ConstellationResNet
from models.vision_transformer_model import ConstellationVisionTransformer
from loaders.constellation_loader import ConstellationDataset
from utils.device_utils import get_device
from utils.loss_utils import WeightedSNRLoss, KendallUncertaintyWeighting, SNRRegressionLoss
from training_constellation import train
import argparse
import warnings
import os
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message=r".*NNPACK.*")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(checkpoint=None, batch_size=1024, snr_list=None, mods_to_process=None, epochs=100, base_lr=0.0001, max_lr=0.001, weight_decay=1e-4, test_size=0.2, model_type="resnet"):
    # Load data
    print("Loading data...")

    image_type = 'grayscale'
    root_dir = "constellation"
    torch.random.manual_seed(42)

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
    dataset = ConstellationDataset(
        root_dir=root_dir,
        image_type=image_type,
        snr_list=snr_list,
        mods_to_process=mods_to_process
    )

    # Print dataset information
    print("\nDataset Information:")
    print("===================")
    print("Modulation Classes:")
    for idx, mod in enumerate(dataset.modulation_labels.keys()):
        print(f"  {idx}: {mod}")
    print("\nSNR Values:")
    print("  ", list(dataset.snr_labels.keys()))
    print("===================\n")

    # Split dataset into train and validation sets
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Add fixed seed for reproducibility
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=12,  # Increased for RTX 4090
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,  # Increased prefetch
        drop_last=True  # Slightly faster and more stable training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size*2,  # Doubled for faster validation
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )

    # Determine input channels based on image_type
    input_channels = 1 if image_type == 'grayscale' else 3

    num_modulation_classes = len(dataset.modulation_labels)
    num_snr_classes = len(dataset.snr_labels)

    print(f"Number of modulation classes: {num_modulation_classes}")
    print(f"Number of SNR classes: {num_snr_classes}")

    # Initialize model based on model_type
    if model_type.lower() == "resnet":
        print("Using ResNet model...")
        model = ConstellationResNet(
            num_classes=num_modulation_classes,
            snr_classes=num_snr_classes,
            input_channels=input_channels,
            model_name="resnet18"
        )
    elif model_type.lower() == "transformer":
        print("Using Vision Transformer model...")
        model = ConstellationVisionTransformer(
            num_classes=num_modulation_classes,
            num_snr_classes=num_snr_classes
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'resnet' or 'transformer'.")

    # Determine device (CUDA, MPS, or CPU)
    device = get_device()

    # Move model and loss functions to device
    model = model.to(device)

    # Set CUDA optimizations if available
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN benchmark mode")

    # Initialize loss functions and Kendall uncertainty-based loss weighting
    criterion_modulation = nn.CrossEntropyLoss().to(device)
    
    # Get the SNR values list from the dataset labels
    snr_values = list(dataset.snr_labels.keys())
    print(f"Using SNR values for classification: {snr_values}")
    
    # Initialize weighted SNR loss with the SNR values
    criterion_snr = WeightedSNRLoss(snr_values=snr_values, device=device)
    criterion_dynamic = KendallUncertaintyWeighting(num_tasks=2, device=device)

    # Initialize optimizer with separate learning rates for model and loss weights
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_dynamic.parameters(), 'lr': 1e-3}  # Higher LR for loss weights
    ], lr=base_lr, weight_decay=weight_decay)

    # Calculate step size for CyclicLR - set for 3 complete cycles over all epochs
    epochs_per_cycle = epochs // 3  # 33.33 epochs per cycle
    step_size_up = epochs_per_cycle // 2  # 16.67 epochs for up phase
    print(f"\nCyclicLR Configuration (3 cycles over {epochs} epochs):")
    print(f"Epochs per cycle: {epochs_per_cycle}")
    print(f"Step size up: {step_size_up} epochs")
    print(f"Base LR: {base_lr}")
    print(f"Max LR: {max_lr}")
    print(f"Cycle pattern: 3 triangular2 cycles over all epochs\n")

    # Initialize CyclicLR scheduler with 3 cycles over all epochs
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,  # ~16.67 epochs up
        mode='triangular2',  # Will automatically reduce max_lr by half each cycle
        cycle_momentum=True
    )

    # Train and validate the model
    train(
        model,
        device,
        criterion_modulation,
        criterion_snr,
        criterion_dynamic,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        epochs=epochs,
        save_dir="checkpoints",
        mod_list=mods_to_process,
        snr_list=snr_list,
        base_lr=base_lr,
        max_lr=max_lr,
        weight_decay=weight_decay,
        checkpoint=checkpoint
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for constellation classification')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=128)
    parser.add_argument('--snr_list', type=str, help='Comma-separated list of SNRs to process')
    parser.add_argument('--mods_to_process', type=str, help='Comma-separated list of modulation types to process')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=100)
    parser.add_argument('--base_lr', type=float, help='Base learning rate for the optimizer', default=1e-5)
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate for the optimizer', default=1e-3)
    parser.add_argument('--weight_decay', type=float, help='Weight decay for the optimizer', default=1e-4)
    parser.add_argument('--test_size', type=float, help='Test size for train/validation split', default=0.15)
    parser.add_argument('--model_type', type=str, help='Type of model to use (resnet or transformer)', default='transformer')
    
    args = parser.parse_args()
    
    main(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        epochs=args.epochs,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        model_type=args.model_type
    )
