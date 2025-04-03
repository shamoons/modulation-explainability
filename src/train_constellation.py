# src/train_constellation.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.constellation_model import ConstellationResNet
from models.vision_transformer_model import ConstellationVisionTransformer
from loaders.constellation_loader import ConstellationDataset
from utils.device_utils import get_device
from utils.loss_utils import WeightedSNRLoss, DynamicLossBalancing
from training_constellation import train
import argparse
import warnings
import os

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


def main(checkpoint=None, batch_size=1024, snr_list=None, mods_to_process=None, epochs=100, base_lr=0.0001, max_lr=0.001, weight_decay=1e-4, test_size=0.2, patience=5, model_type="resnet"):
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
            snr_classes=num_snr_classes,
            input_channels=input_channels
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'resnet' or 'transformer'.")

    # If checkpoint is provided, load the existing model state
    if checkpoint is not None and os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("No checkpoint provided, starting training from scratch.")

    # Determine device (CUDA, MPS, or CPU)
    device = get_device()

    # Move model and loss functions to device
    model = model.to(device)

    # Initialize loss functions
    criterion_modulation = nn.CrossEntropyLoss()
    criterion_snr = WeightedSNRLoss(list(dataset.snr_labels.keys()), device=device)
    criterion_dynamic = DynamicLossBalancing(num_tasks=2, device=device)

    # Initialize optimizer with separate learning rates for model and loss weights
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_dynamic.parameters(), 'lr': 1e-3}  # Higher LR for loss weights
    ], lr=base_lr, weight_decay=weight_decay)

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Number of epochs for first restart
        T_mult=2,  # Factor to increase T_0 after each restart
        eta_min=base_lr/10  # Minimum learning rate
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
        dataset,
        batch_size=batch_size,
        test_size=test_size,
        epochs=epochs,
        mod_list=mods_to_process,
        snr_list=snr_list,
        base_lr=base_lr,
        weight_decay=weight_decay,
        patience=patience
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for constellation classification')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=128)
    parser.add_argument('--snr_list', type=str, help='Comma-separated list of SNRs to process')
    parser.add_argument('--mods_to_process', type=str, help='Comma-separated list of modulation types to process')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=100)
    parser.add_argument('--base_lr', type=float, help='Base learning rate for the optimizer', default=0.0001)
    parser.add_argument('--max_lr', type=float, help='Max learning rate for the optimizer (not used with ReduceLROnPlateau)', default=0.001)
    parser.add_argument('--weight_decay', type=float, help='Weight decay for the optimizer', default=1e-4)
    parser.add_argument('--test_size', type=float, help='Test size for train/validation split', default=0.15)
    parser.add_argument('--patience', type=int, help='Number of epochs to wait before reducing', default=5)
    parser.add_argument('--model_type', type=str, help='Type of model to use (resnet or transformer)', default='resnet')
    
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
        patience=args.patience,
        model_type=args.model_type
    )
