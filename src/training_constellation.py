# src/training_constellation.py

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src.validate_constellation import plot_validation_confusion_matrices
from contextlib import nullcontext
import numpy as np


# Add curriculum imports
try:
    from curriculum import CurriculumManager
    CURRICULUM_AVAILABLE = True
except ImportError:
    CURRICULUM_AVAILABLE = False


def train(
    model,
    device,
    criterion_modulation,
    criterion_snr,
    criterion_dynamic,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    epochs=10,
    save_dir="checkpoints",
    mod_list=None,
    snr_list=None,
    base_lr=None,
    max_lr=None,
    weight_decay=None,
    checkpoint=None,
    use_curriculum=False,
    curriculum_patience=5,
    curriculum_stages=None
):
    """
    Train the model with mixed precision support for CUDA devices
    and optional curriculum learning
    """
    print("\nStarting training...")
    os.makedirs(save_dir, exist_ok=True)

    # Create a subdirectory for plots
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create results directory for confusion matrices (at the same level as checkpoints)
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory at {results_dir}")
    
    # Create confusion matrices directory
    confusion_matrices_dir = os.path.join(results_dir, "confusion_matrices")
    if not os.path.exists(confusion_matrices_dir):
        os.makedirs(confusion_matrices_dir)
        print(f"Created confusion matrices directory at {confusion_matrices_dir}")
    
    # Initialize curriculum manager if enabled
    curriculum_manager = None
    if use_curriculum and CURRICULUM_AVAILABLE:
        print("\nInitializing curriculum learning:")
        curriculum_manager = CurriculumManager(
            stages=curriculum_stages,
            patience=curriculum_patience,
            device=device
        )
    
    # Initialize mixed precision training if CUDA is available
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Initialize wandb with resume support and curriculum information
    config = {
        "architecture": model.__class__.__name__,
        "batch_size": train_loader.batch_size,
        "learning_rate": base_lr,
        "max_learning_rate": max_lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "mod_list": mod_list,
        "snr_list": snr_list,
        "use_curriculum": use_curriculum
    }
    
    # Add curriculum-specific config
    if use_curriculum and curriculum_manager:
        config.update({
            "curriculum_patience": curriculum_patience,
            "curriculum_initial_stage": 0,
            "curriculum_initial_snr_list": curriculum_manager.current_snr_list
        })
    
    wandb.init(
        project="constellation-classification",
        config=config,
        resume=True if checkpoint else False
    )
    
    # Get the SNR mapping from dataset
    # Handle both regular and curriculum dataset structures
    dataset_obj = train_loader.dataset
    if hasattr(dataset_obj, 'dataset'):
        # Regular case with train_loader.dataset being a Subset
        snr_labels = dataset_obj.dataset.snr_labels
    else:
        # Curriculum case with train_loader.dataset being a CurriculumAwareDataset
        snr_labels = dataset_obj.snr_labels
        
    snr_values = list(snr_labels.keys())
    print(f"SNR values for classification: {snr_values}")
    
    # Initialize training state
    best_val_loss = float('inf')
    start_epoch = 0
    
    # If checkpoint is provided, load the existing model state
    if checkpoint is not None and os.path.isfile(checkpoint):
        print(f"\nLoading checkpoint from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, weights_only=True)
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load training state but reset epoch counter
            best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            start_epoch = 0  # Always start from epoch 0
            print(f"Loaded model weights from checkpoint (resetting epoch counter to 0)")
            print(f"Previous best validation loss: {best_val_loss:.4f}")
            
            # Load optimizer and scheduler states
            if 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print("Loaded optimizer state")
            if 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                print("Loaded scheduler state")
            
            # Load criterion dynamic state if it exists
            if 'criterion_dynamic_state' in checkpoint_data:
                criterion_dynamic.load_state_dict(checkpoint_data['criterion_dynamic_state'])
                print(f"Loaded dynamic loss weights")
                weights = criterion_dynamic.get_weights()
                print(f"Current task weights - Mod: {weights[0]:.4f}, SNR: {weights[1]:.4f}")
        else:
            model.load_state_dict(checkpoint_data)
            print("Loaded model weights from simple state dict checkpoint")
        print("Checkpoint loaded successfully.\n")
    else:
        print("\nNo checkpoint provided or checkpoint file not found, starting training from scratch.\n")
    
    # Training loop - use start_epoch
    for epoch in range(start_epoch, epochs):
        print(f"\nStarting Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        running_modulation_loss = 0.0
        running_snr_loss = 0.0
        correct_modulation = 0
        correct_snr = 0
        total = 0
        snr_mae = 0.0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, (inputs, modulation_labels, snr_labels) in enumerate(pbar):
            inputs = inputs.to(device)
            modulation_labels = modulation_labels.to(device)
            snr_labels = snr_labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training for CUDA devices
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    modulation_output, snr_output = model(inputs)
                    loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                    loss_snr = criterion_snr(snr_output, snr_labels)
                    total_loss, loss_weights = criterion_dynamic([loss_modulation, loss_snr])
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                modulation_output, snr_output = model(inputs)
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)
                total_loss, loss_weights = criterion_dynamic([loss_modulation, loss_snr])
                
                total_loss.backward()
                optimizer.step()
            
            # Update metrics
            running_loss += total_loss.item()
            running_modulation_loss += loss_modulation.item()
            running_snr_loss += loss_snr.item()
            
            # Calculate modulation accuracy
            pred_modulation = modulation_output.argmax(dim=1)
            correct_modulation += (pred_modulation == modulation_labels).sum().item()
            total += modulation_labels.size(0)
            
            # Calculate SNR metrics (classification)
            pred_snr_class = torch.argmax(snr_output, dim=1)
            correct_snr += (pred_snr_class == snr_labels).sum().item()
            
            # For backward compatibility, calculate MAE using expected SNR values
            # Get expected SNR values using the same method as in WeightedSNRLoss
            pred_snr_values = criterion_snr.scale_to_snr(snr_output).squeeze()
            
            # Safely map SNR indices to values with bounds checking
            true_snr_values = []
            for idx in snr_labels:
                idx_val = idx.item()
                # Check if index is within bounds
                if 0 <= idx_val < len(snr_values):
                    true_snr_values.append(snr_values[idx_val])
                else:
                    # Handle out-of-bounds index by using a default value
                    # (first SNR value) and logging warning
                    if len(snr_values) > 0:
                        print(f"Warning: SNR index {idx_val} out of bounds (max {len(snr_values)-1}). Using default value.")
                        true_snr_values.append(snr_values[0])
                    else:
                        print(f"Error: SNR values list is empty. Using 0 as default.")
                        true_snr_values.append(0)
            
            true_snr_values = torch.tensor(true_snr_values, device=device, dtype=torch.float32)
            snr_mae += torch.abs(pred_snr_values - true_snr_values).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (i + 1)
            avg_modulation_loss = running_modulation_loss / (i + 1)
            avg_snr_loss = running_snr_loss / (i + 1)
            modulation_accuracy = 100 * correct_modulation / total
            snr_mae_avg = snr_mae / total
            snr_accuracy = 100 * correct_snr / total
            
            # Add curriculum info to progress bar if applicable
            if use_curriculum and curriculum_manager:
                pbar.set_postfix({
                    'Loss(Mod/SNR)': f'{avg_loss:.3f}({avg_modulation_loss:.3f}/{avg_snr_loss:.3f})',
                    'Acc(Mod/SNR)': f'{modulation_accuracy:.2f}%/{snr_accuracy:.2f}%',
                    'SNR MAE': f'{snr_mae_avg:.2f}dB',
                    'W(Mod/SNR)': f'{loss_weights[0]:.2f}/{loss_weights[1]:.2f}',
                    'Curr Stage': f'{curriculum_manager.current_stage}'
                })
            else:
                pbar.set_postfix({
                    'Loss(Mod/SNR)': f'{avg_loss:.3f}({avg_modulation_loss:.3f}/{avg_snr_loss:.3f})',
                    'Acc(Mod/SNR)': f'{modulation_accuracy:.2f}%/{snr_accuracy:.2f}%',
                    'SNR MAE': f'{snr_mae_avg:.2f}dB',
                    'W(Mod/SNR)': f'{loss_weights[0]:.2f}/{loss_weights[1]:.2f}'
                })
        
        # Calculate average metrics for the epoch
        avg_loss = running_loss / len(train_loader)
        modulation_accuracy = 100 * correct_modulation / total
        snr_mae_avg = snr_mae / total
        snr_accuracy = 100 * correct_snr / total
        avg_modulation_loss = running_modulation_loss / len(train_loader)
        avg_snr_loss = running_snr_loss / len(train_loader)
        
        # Validate and get metrics
        print(f"\nValidating after epoch {epoch}...")
        metrics = validate(model, val_loader, device, criterion_dynamic, use_curriculum, curriculum_manager, save_dir, epoch)
        
        # Extract metrics
        val_loss = metrics['val_loss']
        val_modulation_accuracy = metrics['val_modulation_accuracy']
        val_snr_accuracy = metrics['val_snr_accuracy']
        
        # Extract or set default values for other metrics
        val_snr_mae = metrics.get('val_snr_mae', 0.0)
        val_avg_modulation_loss = metrics.get('val_modulation_loss', 0.0)
        val_avg_snr_loss = metrics.get('val_snr_loss', 0.0)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training - Loss: {avg_loss:.4f}, Mod Acc: {modulation_accuracy:.2f}%, SNR MAE: {snr_mae_avg:.2f} dB, SNR Acc: {snr_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Mod Acc: {val_modulation_accuracy:.2f}%, SNR MAE: {val_snr_mae:.2f} dB, SNR Acc: {val_snr_accuracy:.2f}%")
        print(f"Training Losses - Mod: {avg_modulation_loss:.4f}, SNR: {avg_snr_loss:.4f}")
        print(f"Validation Losses - Mod: {val_avg_modulation_loss:.4f}, SNR: {val_avg_snr_loss:.4f}")
        print(f"Task Weights - Modulation: {loss_weights[0]:.4f}, SNR: {loss_weights[1]:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Add curriculum status if enabled
        if use_curriculum and curriculum_manager:
            print(f"Curriculum Stage: {curriculum_manager.current_stage + 1}/{len(curriculum_stages)}")
            print(f"Current SNR List: {curriculum_manager.current_snr_list}")
            print(f"Epochs without SNR accuracy improvement: {curriculum_manager.epochs_without_improvement}/{curriculum_patience}")
        
        print("\nLogging metrics to wandb...")
        # Prepare metrics dict for wandb
        metrics_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_modulation_accuracy": modulation_accuracy,
            "train_snr_mae": snr_mae_avg,
            "train_snr_accuracy": snr_accuracy,
            "train_modulation_loss": avg_modulation_loss,
            "train_snr_loss": avg_snr_loss,
            "val_loss": val_loss,
            "val_modulation_accuracy": val_modulation_accuracy,
            "val_snr_mae": val_snr_mae,
            "val_snr_accuracy": val_snr_accuracy,
            "val_modulation_loss": val_avg_modulation_loss,
            "val_snr_loss": val_avg_snr_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        # Add curriculum metrics if enabled
        if use_curriculum and curriculum_manager:
            metrics_dict.update({
                "curriculum_stage": curriculum_manager.current_stage,
                "curriculum_snr_list": curriculum_manager.current_snr_list,
                "curriculum_epochs_without_improvement": curriculum_manager.epochs_without_improvement
            })
        
        # Log metrics to wandb
        wandb.log(metrics_dict)
        
        # Check for best model
        print("Checking for best model...")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, 'best_model.pth')
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'criterion_dynamic_state': criterion_dynamic.state_dict()
            }
            
            # Save curriculum state if enabled
            if use_curriculum and curriculum_manager:
                checkpoint_data['curriculum_stage'] = curriculum_manager.current_stage
                checkpoint_data['curriculum_snr_list'] = curriculum_manager.current_snr_list
                checkpoint_data['curriculum_history'] = curriculum_manager.stage_history
            
            torch.save(checkpoint_data, model_path, _use_new_zipfile_serialization=True)
            print(f"Saved new best model to {model_path} with validation loss: {best_val_loss:.4f}")
        
        # Curriculum stage progression check
        if use_curriculum and curriculum_manager:
            # Check if we should progress to the next curriculum stage
            if curriculum_manager.should_progress(val_snr_accuracy):
                next_snr_list = curriculum_manager.get_next_stage()
                if next_snr_list:
                    print(f"\n{'#'*70}")
                    print(f"# CURRICULUM ADVANCEMENT: Stage {curriculum_manager.current_stage}")
                    print(f"# New SNR list: {next_snr_list}")
                    print(f"{'#'*70}")
                    
                    # Get dataset sizes before update
                    train_size_before = len(train_loader.dataset)
                    val_size_before = len(val_loader.dataset)
                    
                    print(f"\nCurrent dataset sizes before update:")
                    print(f"- Train dataset: {train_size_before} samples")
                    print(f"- Validation dataset: {val_size_before} samples")
                    
                    # Update train and validation datasets with the new SNR list
                    print(f"\nUpdating training dataset...")
                    train_loader.dataset.update_snr_list(next_snr_list)
                    
                    print(f"\nUpdating validation dataset...")
                    val_loader.dataset.update_snr_list(next_snr_list)
                    
                    # Update SNR loss function with new SNR values
                    print(f"\nUpdating SNR loss function...")
                    criterion_snr.update_snr_values(next_snr_list)
                    
                    # Safety check: Verify dataset sizes after update
                    train_size_after = len(train_loader.dataset)
                    val_size_after = len(val_loader.dataset)
                    
                    print(f"\nCurrent dataset sizes after update:")
                    print(f"- Train dataset: {train_size_after} samples")
                    print(f"- Validation dataset: {val_size_after} samples")
                    
                    # Verify datasets were updated properly
                    if train_size_after == 0 or val_size_after == 0:
                        raise ValueError(f"ERROR: Dataset update resulted in empty dataset! "
                                        f"Train size: {train_size_after}, "
                                        f"Val size: {val_size_after}")
                    
                    # Ensure SNR indices are synchronized with the new SNR list
                    print("\nValidating SNR indices in datasets...")
                    try:
                        # Get current SNR values after update
                        if hasattr(train_loader.dataset, 'snr_labels'):
                            current_snr_values = list(train_loader.dataset.snr_labels.keys())
                        elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'snr_labels'):
                            current_snr_values = list(train_loader.dataset.dataset.snr_labels.keys())
                        else:
                            # No direct access to labels, try using curriculum values
                            current_snr_values = next_snr_list
                        
                        # Verify values match curriculum stages
                        values_match = set(current_snr_values) == set(next_snr_list)
                        if not values_match:
                            print(f"WARNING: SNR values in dataset ({current_snr_values}) don't match curriculum values ({next_snr_list})")
                        else:
                            print(f"SNR values correctly updated to: {current_snr_values}")
                        
                        # Check indices mapping to ensure they're within bounds
                        print(f"New SNR labels mapping: {[i for i in range(len(next_snr_list))]} → {next_snr_list}")
                    except Exception as e:
                        print(f"Warning: Could not validate SNR indices: {str(e)}")
                    
                    # IMPORTANT: Recreate DataLoaders with updated datasets to avoid index errors
                    # Save original DataLoader parameters
                    train_batch_size = train_loader.batch_size
                    val_batch_size = val_loader.batch_size
                    num_workers = train_loader.num_workers
                    
                    # Properly close existing workers before recreating DataLoaders
                    try:
                        train_loader._iterator = None
                        val_loader._iterator = None
                    except:
                        print("Warning: Could not cleanly reset DataLoader iterators")
                    
                    print(f"\nRecreating DataLoaders with updated datasets...")
                    train_loader = DataLoader(
                        train_loader.dataset, 
                        batch_size=train_batch_size, 
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=False,  # Set to False first time
                        prefetch_factor=3,
                        drop_last=True
                    )
                    
                    val_loader = DataLoader(
                        val_loader.dataset, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=False,  # Set to False first time
                        prefetch_factor=3
                    )
                    
                    # Recreate again with persistent workers now that we have clean workers
                    train_loader = DataLoader(
                        train_loader.dataset, 
                        batch_size=train_batch_size, 
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=3,
                        drop_last=True
                    )
                    
                    val_loader = DataLoader(
                        val_loader.dataset, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=3
                    )
                    
                    print(f"\nDatasets updated with new SNR list:")
                    print(f"- Train dataset: {len(train_loader.dataset)} samples")
                    print(f"- Validation dataset: {len(val_loader.dataset)} samples")
                    
                    # Get the actual SNR values in the datasets for verification
                    train_snrs = set()
                    val_snrs = set()
                    
                    # Extract SNR values from datasets
                    if hasattr(train_loader.dataset, 'get_actual_snr_values'):
                        for idx in range(len(train_loader.dataset.snr_labels_list)):
                            snr_idx = train_loader.dataset.snr_labels_list[idx]
                            train_snrs.add(train_loader.dataset.get_actual_snr_values(snr_idx))
                            
                        for idx in range(len(val_loader.dataset.snr_labels_list)):
                            snr_idx = val_loader.dataset.snr_labels_list[idx]
                            val_snrs.add(val_loader.dataset.get_actual_snr_values(snr_idx))
                    
                    print(f"- Training SNR values present: {sorted(list(train_snrs))}")
                    print(f"- Validation SNR values present: {sorted(list(val_snrs))}")
                    
                    # Check if datasets contain all the requested SNR values
                    missing_train = set(next_snr_list) - train_snrs
                    missing_val = set(next_snr_list) - val_snrs
                    
                    if missing_train:
                        print(f"WARNING: Training dataset is missing SNR values: {missing_train}")
                    if missing_val:
                        print(f"WARNING: Validation dataset is missing SNR values: {missing_val}")
                    
                    if not missing_train and not missing_val:
                        print(f"SUCCESS: All SNR values are present in both datasets")
                    
                    # Log stage transition to wandb
                    wandb.log({
                        "curriculum_stage_transition": curriculum_manager.current_stage,
                        "curriculum_new_snr_list": next_snr_list,
                        "curriculum_train_size": len(train_loader.dataset),
                        "curriculum_val_size": len(val_loader.dataset),
                        "curriculum_best_snr_accuracy": curriculum_manager.best_snr_accuracy.item()
                    })
        
        print("Updating learning rate scheduler...")
        # Update learning rate scheduler
        scheduler.step()
        
        print("Gathering validation predictions for plotting...")
        # Pre-allocate tensors for predictions on CPU to avoid memory issues
        val_size = len(val_loader.dataset)
        all_pred_modulation = torch.empty(val_size, dtype=torch.long)
        all_true_modulation = torch.empty(val_size, dtype=torch.long)
        all_pred_snr = torch.empty(val_size, dtype=torch.float32)
        all_true_snr = torch.empty(val_size, dtype=torch.float32)

        model.eval()
        current_idx = 0

        # Validation prediction gathering with device-specific handling
        with torch.no_grad():
            if device.type == 'cuda':
                context = torch.amp.autocast('cuda')
            else:
                context = nullcontext()
            
            with context:
                for inputs, modulation_targets, snr_targets in tqdm(val_loader, desc="Processing validation batches"):
                    batch_size = inputs.size(0)
                    
                    # Move to device
                    inputs = inputs.to(device)
                    modulation_targets = modulation_targets.to(device)
                    snr_targets = snr_targets.to(device)
                    
                    # Get model predictions
                    modulation_output, snr_output = model(inputs)
                    
                    # Get predicted modulation and SNR
                    predicted_modulation = modulation_output.argmax(dim=1)
                    
                    # Get SNR values using expected value method from the loss function
                    pred_snr_values = criterion_snr.scale_to_snr(snr_output).squeeze()
                    
                    # Safely map SNR indices to values with bounds checking
                    true_snr_values = []
                    for idx in snr_targets:
                        idx_val = idx.item()
                        # Check if index is within bounds
                        if 0 <= idx_val < len(snr_values):
                            true_snr_values.append(snr_values[idx_val])
                        else:
                            # Handle out-of-bounds index by using a default value
                            # (first SNR value) and logging warning
                            if len(snr_values) > 0:
                                print(f"Warning: SNR index {idx_val} out of bounds (max {len(snr_values)-1}). Using default value.")
                                true_snr_values.append(snr_values[0])
                            else:
                                print(f"Error: SNR values list is empty. Using 0 as default.")
                                true_snr_values.append(0)
                    
                    true_snr_values = torch.tensor(true_snr_values, device=device, dtype=torch.float32)
                    
                    # Store in pre-allocated tensors (move to CPU in batches)
                    slice_idx = slice(current_idx, current_idx + batch_size)
                    all_pred_modulation[slice_idx] = predicted_modulation.cpu()
                    all_true_modulation[slice_idx] = modulation_targets.cpu()
                    all_pred_snr[slice_idx] = pred_snr_values.cpu()
                    all_true_snr[slice_idx] = true_snr_values.cpu()
                    
                    current_idx += batch_size

        # Convert to numpy arrays once at the end
        all_pred_modulation = all_pred_modulation.numpy()
        all_true_modulation = all_true_modulation.numpy()
        all_pred_snr = all_pred_snr.numpy()
        all_true_snr = all_true_snr.numpy()
        
        print("Plotting confusion matrices...")
        # Get the modulation class names using our helper function
        # Add the helper function if it doesn't exist already
        if not 'get_dataset_property' in globals():
            def get_dataset_property(dataset_obj, property_name):
                """Helper to access dataset properties consistently"""
                if hasattr(dataset_obj, 'dataset'):
                    # Regular case with dataset_obj being a Subset
                    return getattr(dataset_obj.dataset, property_name)
                else:
                    # Curriculum case with dataset_obj being a direct dataset
                    return getattr(dataset_obj, property_name)
        
        # Plot and save confusion matrices
        modulation_class_names = list(get_dataset_property(train_loader.dataset, 'modulation_labels').keys())
        
        # Debug - print SNR accuracy for comparison with confusion matrix
        print(f"\nDebug - Overall validation SNR accuracy: {val_snr_accuracy:.2f}%")
        print(f"This should roughly match the confusion matrix visualization\n")
        
        plot_validation_confusion_matrices(
            true_modulation=all_true_modulation,
            pred_modulation=all_pred_modulation,
            true_snr_indices=all_true_snr,
            pred_snr_indices=all_pred_snr,
            mod_classes=modulation_class_names, 
            save_dir=confusion_matrices_dir,
            epoch=epoch+1,
            use_curriculum=use_curriculum,
            current_snr_list=curriculum_manager.get_current_snr_list() if use_curriculum and curriculum_manager else None,
            metrics={'val_dataset': val_loader.dataset}
        )
        print(f"Saved confusion matrices to {confusion_matrices_dir}")
    
    # Close wandb
    wandb.finish()
    
    return os.path.join(save_dir, 'best_model.pth')

def validate_constellation(model, val_dataloader, criterion, device, 
                            use_curriculum=False, curriculum_manager=None, 
                            save_results_dir=None, epoch=None, 
                            visualize=False, mod_classes=None):
    """
    Validate a constellation recognition model
    
    Args:
        model: The PyTorch model to validate
        val_dataloader: Validation data loader
        criterion: Loss criterion
        device: Device to run validation on
        use_curriculum: Whether to use curriculum learning
        curriculum_manager: CurriculumManager instance if using curriculum
        save_results_dir: Directory to save validation results
        epoch: Current epoch number
        visualize: Whether to visualize validation results
        mod_classes: List of modulation class names for visualization
    
    Returns:
        val_loss, mod_accuracy, snr_accuracy, metrics_dict
    """
    # Set model to evaluation mode
    model.eval()
    
    val_loss = 0.0
    correct_mod = 0
    correct_snr = 0
    total = 0
    
    # Store ground truth and predictions for visualization
    true_mod_labels = []
    pred_mod_labels = []
    true_snr_indices = []
    pred_snr_indices = []
    
    # For curriculum mode - track per-SNR accuracy and record a direct confusion matrix
    metrics = {}
    
    # Get the SNR values from the dataset for proper mapping
    try:
        if hasattr(val_dataloader.dataset, 'dataset'):
            dataset_snr_labels = val_dataloader.dataset.dataset.snr_labels
        else:
            dataset_snr_labels = val_dataloader.dataset.snr_labels
            
        snr_values = list(dataset_snr_labels.keys())
        metrics['snr_values'] = snr_values
        print(f"Available SNR values for validation: {snr_values}")
    except Exception as e:
        print(f"Error getting SNR values from dataset: {str(e)}")
        snr_values = []
        
    # Use curriculum SNR values if available
    if use_curriculum and curriculum_manager is not None:
        # Get current SNR values
        current_snr_list = curriculum_manager.get_current_snr_list()
        metrics['current_snr_list'] = current_snr_list
        print(f"Current curriculum stage: {curriculum_manager.current_stage}")
        print(f"Current SNR list: {current_snr_list}")
        print(f"Length of current SNR list: {len(current_snr_list)}")
        
        # Initialize confusion matrix for curriculum SNRs
        n_curr_classes = len(current_snr_list)
        curriculum_cm = np.zeros((n_curr_classes, n_curr_classes))
        
        # Store mapping from SNR values to their curriculum indices
        snr_to_curr_idx = {snr: i for i, snr in enumerate(sorted(current_snr_list))}
        metrics['curriculum_snr_values'] = sorted(current_snr_list)
        
    # Disable gradient computation for validation
    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            inputs, mod_labels, snr_indices = \
                data[0].to(device), data[1].to(device), data[2].to(device)
                
            # Get actual SNRs from the dataset for curriculum
            if use_curriculum and hasattr(val_dataloader.dataset, 'get_actual_snr_values'):
                batch_snr_values = []
                for i in range(len(snr_indices)):
                    snr_idx = snr_indices[i].item()
                    snr_value = val_dataloader.dataset.get_actual_snr_values(snr_idx)
                    batch_snr_values.append(snr_value)
            
            # Forward pass
            mod_out, snr_out = model(inputs)
            
            # Calculate loss
            mod_loss = criterion(mod_out, mod_labels)
            snr_loss = criterion(snr_out, snr_indices)
            loss = mod_loss + snr_loss
            
            # Update validation loss
            val_loss += loss.item()
            
            # Get predictions
            _, mod_preds = torch.max(mod_out, 1)
            _, snr_preds = torch.max(snr_out, 1)
            
            # Update accuracy metrics
            batch_size = mod_labels.size(0)
            total += batch_size
            correct_mod += (mod_preds == mod_labels).sum().item()
            correct_snr += (snr_preds == snr_indices).sum().item()
            
            # Store for visualization
            true_mod_labels.extend(mod_labels.cpu().numpy())
            pred_mod_labels.extend(mod_preds.cpu().numpy())
            true_snr_indices.extend(snr_indices.cpu().numpy())
            pred_snr_indices.extend(snr_preds.cpu().numpy())
            
            # Track per-SNR accuracy for curriculum mode
            if use_curriculum and curriculum_manager is not None:
                # For each sample in batch, update the confusion matrix
                for i in range(batch_size):
                    try:
                        # Get the actual SNR value for this sample
                        if hasattr(val_dataloader.dataset, 'get_actual_snr_values'):
                            true_snr_idx = snr_indices[i].item()
                            pred_snr_idx = snr_preds[i].item()
                            
                            # Safely convert indices to SNR values with bounds checking
                            try:
                                true_snr_val = val_dataloader.dataset.get_actual_snr_values(true_snr_idx)
                                pred_snr_val = val_dataloader.dataset.get_actual_snr_values(pred_snr_idx)
                                
                                # Check if these SNRs are in our current curriculum
                                if true_snr_val in snr_to_curr_idx and pred_snr_val in snr_to_curr_idx:
                                    true_curr_idx = snr_to_curr_idx[true_snr_val]
                                    pred_curr_idx = snr_to_curr_idx[pred_snr_val]
                                    curriculum_cm[true_curr_idx, pred_curr_idx] += 1
                            except (IndexError, ValueError) as e:
                                # Silently continue if conversion fails
                                # We don't want to pollute logs with many errors
                                pass
                    except Exception as e:
                        # Catch broader exceptions but don't halt validation
                        continue
    
    # Calculate average metrics
    val_loss /= len(val_dataloader)
    mod_accuracy = 100 * correct_mod / total
    snr_accuracy = 100 * correct_snr / total
    
    # Store metrics for visualization
    metrics['mod_accuracy'] = mod_accuracy
    metrics['snr_accuracy'] = snr_accuracy
    
    if use_curriculum and curriculum_manager is not None:
        # Add curriculum metrics
        metrics['curriculum_cm'] = curriculum_cm
        print(f"\nDirect Curriculum SNR Confusion Matrix:")
        print(curriculum_cm)
        
        # Calculate overall accuracy from confusion matrix for verification
        cm_diagonal_sum = np.sum(np.diag(curriculum_cm))
        cm_total = np.sum(curriculum_cm)
        cm_accuracy = 100 * cm_diagonal_sum / cm_total if cm_total > 0 else 0
        print(f"Overall SNR accuracy from confusion matrix: {cm_accuracy:.2f}%")
        print(f"Overall SNR accuracy from predictions: {snr_accuracy:.2f}%")
        metrics['cm_snr_accuracy'] = cm_accuracy
    
    # Visualize validation results if requested
    if visualize and save_results_dir:
        # Create visualization directory if it doesn't exist
        os.makedirs(save_results_dir, exist_ok=True)
        
        # Plot confusion matrices
        from src.validate_constellation import plot_validation_confusion_matrices
        
        # Pass metric information for visualization
        plot_validation_confusion_matrices(
            true_modulation=true_mod_labels,
            pred_modulation=pred_mod_labels,
            true_snr_indices=true_snr_indices,
            pred_snr_indices=pred_snr_indices,
            mod_classes=mod_classes,
            save_dir=save_results_dir,
            epoch=epoch,
            use_curriculum=use_curriculum,
            current_snr_list=curriculum_manager.get_current_snr_list() if use_curriculum and curriculum_manager else None,
            metrics=metrics
        )
    
    return val_loss, mod_accuracy, snr_accuracy, metrics

@torch.no_grad()
def validate(model, val_loader, device, criterion_dynamic, use_curriculum, curriculum_manager, save_dir, epoch):
    """
    Validate the model on the validation set with error handling.
    
    Args:
        model: The PyTorch model to validate
        val_loader: Validation data loader
        device: Device for model execution
        criterion_dynamic: Loss function
        use_curriculum: Whether curriculum learning is enabled
        curriculum_manager: Curriculum manager instance
        save_dir: Directory to save validation results
        epoch: Current epoch
        
    Returns:
        Dict with validation metrics
    """
    model.eval()
    
    try:
        print(f"\nValidating epoch {epoch}...")
        # Safety check
        if len(val_loader.dataset) == 0:
            raise ValueError("Validation dataset is empty!")
            
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # Get information about current SNR values
        try:
            if hasattr(val_loader.dataset, 'snr_labels'):
                snr_values = list(val_loader.dataset.snr_labels.keys())
            elif hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'snr_labels'):
                snr_values = list(val_loader.dataset.dataset.snr_labels.keys())
            else:
                snr_values = []
                
            print(f"SNR values in validation dataset: {snr_values}")
            
            if use_curriculum and curriculum_manager:
                curr_snr_list = curriculum_manager.get_current_snr_list()
                print(f"Current curriculum SNR list: {curr_snr_list}")
                
                # Check for mismatches
                if set(snr_values) != set(curr_snr_list):
                    print(f"WARNING: SNR values in dataset don't match curriculum values")
        except Exception as e:
            print(f"Warning: Could not get SNR values from dataset: {str(e)}")
        
        # Validate using standard metrics
        try:
            val_loss, val_modulation_accuracy, val_snr_accuracy, metrics = validate_constellation(
                model, val_loader, criterion_dynamic, device,
                use_curriculum=use_curriculum,
                curriculum_manager=curriculum_manager,
                save_results_dir=os.path.join(save_dir, 'validation_results', f'epoch_{epoch}'),
                epoch=epoch
            )
        except IndexError as e:
            print(f"IndexError during validation: {str(e)}")
            print("This is likely due to SNR indices being out of bounds after curriculum change.")
            print("Returning default metrics and continuing training...")
            return {
                'val_loss': float('inf'),
                'val_modulation_accuracy': 0.0,
                'val_snr_accuracy': 0.0,
                'val_snr_mae': 0.0
            }
        
        # Add core metrics to the metrics dictionary if not already present
        if 'val_loss' not in metrics:
            metrics['val_loss'] = val_loss
        if 'val_modulation_accuracy' not in metrics:
            metrics['val_modulation_accuracy'] = val_modulation_accuracy
        if 'val_snr_accuracy' not in metrics:
            metrics['val_snr_accuracy'] = val_snr_accuracy
            
        # Log to wandb
        wandb.log({
            'val_loss': val_loss,
            'val_modulation_accuracy': val_modulation_accuracy,
            'val_snr_accuracy': val_snr_accuracy,
            'epoch': epoch
        })
        
        # Log per-class metrics if available
        for key, value in metrics.items():
            if key.startswith('snr_') and key.endswith('_accuracy'):
                wandb.log({key: value, 'epoch': epoch})
        
        # Print validation results
        print(f"Validation Results (Epoch {epoch}):")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Modulation Accuracy: {val_modulation_accuracy:.2f}%")
        print(f"  SNR Accuracy: {val_snr_accuracy:.2f}%")
        
        return metrics
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        print("Continuing training despite validation error...")
        
        # Return default metrics to continue training
        default_metrics = {
            'val_loss': float('inf'),
            'val_modulation_accuracy': 0.0,
            'val_snr_accuracy': 0.0
        }
        return default_metrics
