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
    Train a constellation recognition model with curriculum learning support.
    
    Args:
        model: PyTorch model to train
        device: Device to train on
        criterion_modulation: Loss function for modulation classification
        criterion_snr: Loss function for SNR classification
        criterion_dynamic: Dynamic loss weighting function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        save_dir: Directory to save checkpoints
        mod_list: List of modulation types
        snr_list: List of SNR values
        base_lr: Base learning rate
        max_lr: Maximum learning rate
        weight_decay: Weight decay for optimizer
        checkpoint: Path to checkpoint to resume from
        use_curriculum: Whether to use curriculum learning
        curriculum_patience: Epochs without improvement before curriculum progression
        curriculum_stages: List of curriculum stages
    """
    # Initialize curriculum manager if using curriculum learning
    curriculum_manager = None
    if use_curriculum and curriculum_stages:
        curriculum_manager = CurriculumManager(
            stages=curriculum_stages,
            patience=curriculum_patience,
            device=device
        )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize training metrics
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct_mod = 0
        correct_snr = 0
        total = 0
        
        for batch_idx, data in enumerate(train_loader):
            inputs, mod_labels, snr_indices = \
                data[0].to(device), data[1].to(device), data[2].to(device)
            
            # Forward pass
            mod_out, snr_out = model(inputs)
            
            # Calculate individual losses
            mod_loss = criterion_modulation(mod_out, mod_labels)
            snr_loss = criterion_snr(snr_out, snr_indices)
            
            # Combine losses using dynamic weighting if available
            if criterion_dynamic is not None:
                loss, _ = criterion_dynamic([mod_loss, snr_loss])
            else:
                loss = mod_loss + snr_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, mod_preds = torch.max(mod_out, 1)
            _, snr_preds = torch.max(snr_out, 1)
            
            total += mod_labels.size(0)
            correct_mod += (mod_preds == mod_labels).sum().item()
            correct_snr += (snr_preds == snr_indices).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - "
                      f"Mod Acc: {100*correct_mod/total:.2f}% - "
                      f"SNR Acc: {100*correct_snr/total:.2f}%")
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_mod_accuracy = 100 * correct_mod / total
        train_snr_accuracy = 100 * correct_snr / total
        
        # Validation phase
        val_loss, val_mod_accuracy, val_snr_accuracy, metrics = validate_constellation(
            model=model,
            val_dataloader=val_loader,
            criterion_modulation=criterion_modulation,
            criterion_snr=criterion_snr,
            criterion_dynamic=criterion_dynamic,
            device=device,
            use_curriculum=use_curriculum,
            curriculum_manager=curriculum_manager,
            save_results_dir=save_dir,
            epoch=epoch,
            visualize=True,
            mod_classes=mod_list
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Mod Accuracy: {train_mod_accuracy:.2f}%")
        print(f"Train SNR Accuracy: {train_snr_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Mod Accuracy: {val_mod_accuracy:.2f}%")
        print(f"Val SNR Accuracy: {val_snr_accuracy:.2f}%")
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_mod_accuracy)
        val_accuracies.append(val_mod_accuracy)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_mod_accuracy,
                'val_accuracy': val_mod_accuracy,
                'curriculum_stage': curriculum_manager.current_stage if curriculum_manager else None
            }, os.path.join(save_dir, 'best_model.pth'))
        
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
                    
                    # Verify SNR values in datasets
                    try:
                        train_snr_values = list(train_loader.dataset.snr_labels.keys())
                        val_snr_values = list(val_loader.dataset.snr_labels.keys())
                        print(f"\nSNR values in datasets after update:")
                        print(f"- Train dataset SNRs: {train_snr_values}")
                        print(f"- Validation dataset SNRs: {val_snr_values}")
                        
                        # Verify that all SNRs in next_snr_list are present
                        missing_train = set(next_snr_list) - set(train_snr_values)
                        missing_val = set(next_snr_list) - set(val_snr_values)
                        
                        if missing_train:
                            print(f"WARNING: Missing SNRs in training dataset: {missing_train}")
                        if missing_val:
                            print(f"WARNING: Missing SNRs in validation dataset: {missing_val}")
                            
                    except Exception as e:
                        print(f"Error verifying SNR values: {str(e)}")
                    
                    # Reset optimizer and scheduler for new stage
                    if scheduler is not None:
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=max_lr,
                            epochs=epochs,
                            steps_per_epoch=len(train_loader),
                            pct_start=0.3,
                            div_factor=25.0,
                            final_div_factor=1000.0
                        )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy
    }

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
