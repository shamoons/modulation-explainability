# src/train_constellation.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.constellation_model import ConstellationResNet
from models.vision_transformer_model import ConstellationVisionTransformer
from models.swin_transformer_model import ConstellationSwinTransformer
from loaders.constellation_loader import ConstellationDataset
from utils.device_utils import get_device
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


def main(checkpoint=None, batch_size=32, snr_list=None, mods_to_process=None, epochs=50, base_lr=1e-4, weight_decay=1e-5, test_size=0.2, patience=10, model_type="resnet18", dropout=0.2, use_task_specific=False, use_dilated_preprocessing=False, max_lr=None, step_size_up=5, step_size_down=5, snr_alpha=0.5):
    # Load data
    print("Loading data...")

    image_type = 'grayscale'
    root_dir = "constellation_diagrams"
    torch.random.manual_seed(42)

    # Parse snr_list and mods_to_process if provided
    if snr_list is not None:
        snr_list = [int(s.strip()) for s in snr_list.split(',')]
    else:
        snr_list = None  # Load all SNRs

    if mods_to_process is not None:
        mods_to_process = [mod.strip() for mod in mods_to_process.split(',')]
    else:
        # Default: exclude analog modulations (AM, FM, GMSK, OOK)
        analog_mods = ['AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC', 'FM', 'GMSK', 'OOK']
        # Get all available modulations from constellation directory
        import os
        all_mods = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        mods_to_process = [mod for mod in all_mods if mod not in analog_mods]
        print(f"Using digital modulations only: {sorted(mods_to_process)}")
        print(f"Excluded analog modulations: {sorted(analog_mods)}")

    # Load full dataset (with modulation types and SNRs filtering)
    dataset = ConstellationDataset(
        root_dir=root_dir,
        image_type=image_type,
        snr_list=snr_list,
        mods_to_process=mods_to_process
    )

    # Determine input channels based on image_type
    input_channels = 1 if image_type == 'grayscale' else 3

    num_snr_classes = len(dataset.snr_labels)
    num_modulation_classes = len(dataset.modulation_labels)

    print(f"Number of modulation classes: {num_modulation_classes}")
    print(f"Number of SNR classes: {num_snr_classes}")

    # Create model based on model_type
    if model_type in ["resnet18", "resnet34"]:
        model = ConstellationResNet(
            num_classes=num_modulation_classes,
            snr_classes=num_snr_classes,
            input_channels=input_channels,
            dropout_prob=dropout,
            model_name=model_type
        )
        print(f"Using model: {model_type}")
    elif model_type in ["vit_b_16", "vit_b_32", "vit_h_14"]:
        # Extract patch size from model type
        if model_type == "vit_h_14":
            patch_size = 14
        elif model_type == "vit_b_16":
            patch_size = 16
        else:  # vit_b_32
            patch_size = 32
        model = ConstellationVisionTransformer(
            num_classes=num_modulation_classes,
            snr_classes=num_snr_classes,
            input_channels=input_channels,
            dropout_prob=dropout,
            patch_size=patch_size
        )
        print(f"Using model: {model_type} (patch_size={patch_size})")
    elif model_type in ["swin_tiny", "swin_small", "swin_base"]:
        model = ConstellationSwinTransformer(
            num_classes=num_modulation_classes,
            snr_classes=num_snr_classes,
            input_channels=input_channels,
            dropout_prob=dropout,
            model_variant=model_type,
            use_task_specific=use_task_specific,
            use_dilated_preprocessing=use_dilated_preprocessing
        )
        task_specific_status = "with task-specific extraction" if use_task_specific else "without task-specific extraction"
        dilated_status = "with dilated preprocessing" if use_dilated_preprocessing else "without dilated preprocessing"
        print(f"Using model: {model_type} (Swin Transformer, {task_specific_status}, {dilated_status})")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from: resnet18, resnet34, vit_b_16, vit_b_32, vit_h_14, swin_tiny, swin_small, swin_base")

    # If checkpoint is provided, load the existing model state
    if checkpoint is not None and os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("No checkpoint provided, starting training from scratch.")

    # Get device first
    device = get_device()
    
    # Initialize loss functions
    criterion_modulation = nn.CrossEntropyLoss().to(device)  # Modulation classification loss
    
    # Use classification loss with distance penalty for SNR
    from losses.distance_weighted_ce import DistanceWeightedCrossEntropyLoss
    criterion_snr = DistanceWeightedCrossEntropyLoss(num_classes=num_snr_classes, alpha=snr_alpha).to(device)
    print(f"Using distance-weighted SNR loss with alpha={snr_alpha} (higher = stronger distance penalty)")
    
    # Initialize analytical uncertainty weighting for multi-task learning
    from losses.uncertainty_weighted_loss import AnalyticalUncertaintyWeightedLoss
    uncertainty_weighter = AnalyticalUncertaintyWeightedLoss(num_tasks=2, temperature=1.5, device=device, min_weight=0.05)

    # Initialize optimizer (include uncertainty weighter parameters)
    model_params = list(model.parameters()) + list(uncertainty_weighter.parameters())
    optimizer = optim.Adam(model_params, lr=base_lr, weight_decay=weight_decay)

    # Scheduler will be created in the train function after we know the train_loader size
    scheduler = None  # Will be created in train function

    # Determine device (CUDA, MPS, or CPU)
    device = get_device()

    # Train and validate the model
    # IMPORTANT: Ensure that in your train() function in training_constellation.py, after computing val_loss each epoch,
    # you call: scheduler.step(val_loss)
    train(
        model,
        device,
        criterion_modulation,
        criterion_snr,
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
        patience=patience,
        uncertainty_weighter=uncertainty_weighter,  # Pass the uncertainty weighter
        model_type=model_type,
        dropout=dropout,
        max_lr=max_lr,
        step_size_up=step_size_up,
        step_size_down=step_size_down,
        snr_alpha=snr_alpha
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train constellation model with optional checkpoint loading')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing model checkpoint to resume training', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation', default=32)
    parser.add_argument('--snr_list', type=str, help='Comma-separated list of SNR values to load (default: all SNRs)', default=None)
    parser.add_argument('--mods_to_process', type=str, help='Comma-separated list of modulation types to load (default: all modulations)', default=None)
    parser.add_argument('--epochs', type=int, help='Total number of epochs for training', default=100)
    parser.add_argument('--base_lr', type=float, help='Base learning rate for the optimizer', default=1e-4)
    parser.add_argument('--weight_decay', type=float, help='Weight decay for the optimizer', default=1e-5)
    parser.add_argument('--test_size', type=float, help='Test size for train/validation split', default=0.2)
    parser.add_argument('--patience', type=int, help='Number of epochs to wait before reducing LR', default=10)
    parser.add_argument('--model_type', type=str, help='Model architecture to use', default='resnet18', choices=['resnet18', 'resnet34', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'swin_tiny', 'swin_small', 'swin_base'])
    parser.add_argument('--dropout', type=float, help='Dropout rate for model regularization', default=0.3)
    parser.add_argument('--use_task_specific', type=str2bool, help='Use task-specific feature extraction (Swin only)', default=False)
    parser.add_argument('--use_dilated_preprocessing', type=str2bool, help='Use dilated CNN preprocessing for global context (Swin only)', default=False)
    
    # Cyclic learning rate scheduler options
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate for cyclic scheduler (default: 50x base_lr)', default=None)
    parser.add_argument('--step_size_up', type=int, help='Number of epochs for upward LR cycle', default=5)
    parser.add_argument('--step_size_down', type=int, help='Number of epochs for downward LR cycle', default=5)
    
    # SNR loss options
    parser.add_argument('--snr_alpha', type=float, help='Weight for SNR distance penalty (0=pure CE, 1=strong penalty, 2=very strong)', default=0.5)

    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        epochs=args.epochs,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        test_size=args.test_size,
        model_type=args.model_type,
        dropout=args.dropout,
        use_task_specific=args.use_task_specific,
        use_dilated_preprocessing=args.use_dilated_preprocessing,
        max_lr=args.max_lr,
        step_size_up=args.step_size_up,
        step_size_down=args.step_size_down,
        snr_alpha=args.snr_alpha
    )
