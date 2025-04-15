# src/train_constellation.py

import torch
import torch.nn as nn
import torch.optim as optim
from models.constellation_model import ConstellationResNet
from models.vision_transformer_model import ConstellationVisionTransformer
from models.swin_constellation import SwinConstellation
from loaders.constellation_loader import ConstellationDataset
from utils.device_utils import get_device
from utils.loss_utils import WeightedSNRLoss, KendallUncertaintyWeighting
from training_constellation import train
import argparse
import warnings
import sys
from torch.utils.data import DataLoader, random_split

# Add curriculum learning imports
try:
    from curriculum import CurriculumAwareDataset, CURRICULUM_STAGES, DEFAULT_CURRICULUM_PATIENCE
    CURRICULUM_AVAILABLE = True
except ImportError:
    print("Curriculum learning modules not found, disabling curriculum functionality")
    CURRICULUM_AVAILABLE = False

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


def main(checkpoint=None, batch_size=1024, snr_list=None, mods_to_process=None, 
         epochs=100, base_lr=0.0001, max_lr=0.001, weight_decay=1e-4, test_size=0.2, 
         model_type="resnet", use_curriculum=False, curriculum_patience=DEFAULT_CURRICULUM_PATIENCE):
    # Load data
    print("Loading data...")

    # image_type = 'point'
    # root_dir = "constellation_points"
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

    # Import and use curriculum dataset if available and requested
    use_curriculum = use_curriculum and CURRICULUM_AVAILABLE
    if use_curriculum:
        try:
            from curriculum import CurriculumAwareDataset, CURRICULUM_STAGES
            print("\nInitializing curriculum learning mode:")
            print(f"Patience: {curriculum_patience} epochs")
            
            # Important: Start with only the first stage's SNR values
            initial_snr_list = CURRICULUM_STAGES[0]['snr_list'] 
            print(f"Starting with SNR values: {initial_snr_list}")
            print(f"Total stages: {len(CURRICULUM_STAGES)}")
            
            # Override snr_list with initial curriculum stage
            snr_list = initial_snr_list
            
            # Use curriculum-aware dataset
            dataset_class = CurriculumAwareDataset
        except ImportError:
            print("Curriculum learning modules not available. Falling back to standard training.")
            use_curriculum = False
            dataset_class = ConstellationDataset
    else:
        # Use standard dataset
        dataset_class = ConstellationDataset

    # Load dataset with appropriate class
    dataset = dataset_class(
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
    
    # Create initial splits
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # If using curriculum, we need to ensure both datasets are curriculum-aware
    if use_curriculum:
        # Create new curriculum-aware datasets from the splits
        # We need to do this because random_split returns dataset views, not the original dataset class
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        
        # Create curriculum-aware datasets with the same indices
        train_dataset = dataset_class(
            root_dir=root_dir,
            image_type=image_type,
            snr_list=snr_list,
            mods_to_process=mods_to_process
        )
        val_dataset = dataset_class(
            root_dir=root_dir,
            image_type=image_type,
            snr_list=snr_list,
            mods_to_process=mods_to_process
        )
        
        # Filter datasets to include only the respective indices
        train_dataset.image_paths = [dataset.image_paths[i] for i in train_indices]
        train_dataset.modulation_labels_list = [dataset.modulation_labels_list[i] for i in train_indices]
        train_dataset.snr_labels_list = [dataset.snr_labels_list[i] for i in train_indices]
        
        val_dataset.image_paths = [dataset.image_paths[i] for i in val_indices]
        val_dataset.modulation_labels_list = [dataset.modulation_labels_list[i] for i in val_indices]
        val_dataset.snr_labels_list = [dataset.snr_labels_list[i] for i in val_indices]
        
        # Store original data for filtering
        train_dataset.original_image_paths = train_dataset.image_paths.copy()
        train_dataset.original_modulation_labels_list = train_dataset.modulation_labels_list.copy()
        train_dataset.original_snr_labels_list = train_dataset.snr_labels_list.copy()
        
        val_dataset.original_image_paths = val_dataset.image_paths.copy()
        val_dataset.original_modulation_labels_list = val_dataset.modulation_labels_list.copy()
        val_dataset.original_snr_labels_list = val_dataset.snr_labels_list.copy()
        
        print(f"Created curriculum-aware training dataset with {len(train_dataset)} samples")
        print(f"Created curriculum-aware validation dataset with {len(val_dataset)} samples")
    
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
            model_name="resnet18"
        )
    elif model_type.lower() == "transformer":
        print("Using Vision Transformer model...")
        model = ConstellationVisionTransformer(
            num_classes=num_modulation_classes,
            num_snr_classes=num_snr_classes
        )
    elif model_type.lower() == "swin":
        print("Using Swin Transformer model...")
        model = SwinConstellation(
            num_classes=num_modulation_classes,
            num_snr_classes=num_snr_classes,
            dropout_prob=0.3  # Consistent with VisionTransformer default
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'resnet', 'transformer', or 'swin'.")

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
    if use_curriculum:
        # For curriculum learning, use only the initial stage's SNR values for the loss function
        initial_snr_values = CURRICULUM_STAGES[0]['snr_list']
        print(f"Using initial curriculum SNR values for loss function: {initial_snr_values}")
        criterion_snr = WeightedSNRLoss(snr_values=initial_snr_values, device=device)
    else:
        # For standard training, use all SNR values
        snr_values = list(dataset.snr_labels.keys())
        print(f"Using SNR values for classification: {snr_values}")
        criterion_snr = WeightedSNRLoss(snr_values=snr_values, device=device)
    
    criterion_dynamic = KendallUncertaintyWeighting(num_tasks=2, device=device)

    # Initialize optimizer with separate learning rates for model and loss weights
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_dynamic.parameters(), 'lr': 1e-3}  # Higher LR for loss weights
    ], lr=base_lr, weight_decay=weight_decay)

    # Calculate step size for CyclicLR - set for 5 complete cycles over all epochs
    epochs_per_cycle = epochs // 5  # 20 epochs per cycle
    step_size_up = epochs_per_cycle // 2  # 10 epochs for up phase
    print(f"\nCyclicLR Configuration (5 cycles over {epochs} epochs):")
    print(f"Epochs per cycle: {epochs_per_cycle}")
    print(f"Step size up: {step_size_up} epochs")
    print(f"Base LR: {base_lr}")
    print(f"Max LR: {max_lr}")
    print(f"Cycle pattern: 5 triangular2 cycles over all epochs\n")

    # Initialize CyclicLR scheduler with 5 cycles over all epochs
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,  # 10 epochs up
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
        checkpoint=checkpoint,
        use_curriculum=use_curriculum,
        curriculum_patience=curriculum_patience,
        curriculum_stages=CURRICULUM_STAGES if use_curriculum else None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for constellation classification')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=128)
    parser.add_argument('--snr_list', type=str, help='Comma-separated list of SNRs to process')
    parser.add_argument('--mods_to_process', type=str, help='Comma-separated list of modulation types to process')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=100)
    parser.add_argument('--base_lr', type=float, help='Base learning rate for the optimizer', default=1e-5)
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate for the optimizer', default=1e-4)
    parser.add_argument('--weight_decay', type=float, help='Weight decay for the optimizer', default=1e-4)
    parser.add_argument('--test_size', type=float, help='Test size for train/validation split', default=0.15)
    parser.add_argument('--model_type', type=str, help='Type of model to use (resnet or transformer)', default='transformer')
    
    # Add curriculum learning arguments
    parser.add_argument('--use_curriculum', type=str2bool, help='Enable curriculum learning', default=False)
    parser.add_argument('--curriculum_patience', type=int, help='Epochs without improvement before stage progression', 
                       default=DEFAULT_CURRICULUM_PATIENCE if CURRICULUM_AVAILABLE else 2)
    
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
        model_type=args.model_type,
        use_curriculum=args.use_curriculum,
        curriculum_patience=args.curriculum_patience
    )
