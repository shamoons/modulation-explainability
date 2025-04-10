# src/training_constellation.py

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.amp import GradScaler 
from validate_constellation import plot_validation_confusion_matrices
from contextlib import nullcontext


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
    checkpoint=None
):
    """
    Train the model with dynamic weighted loss.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
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
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb with resume support
    wandb.init(
        project="constellation-classification",
        config={
            "architecture": model.__class__.__name__,
            "batch_size": train_loader.batch_size,
            "learning_rate": base_lr,
            "max_learning_rate": max_lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "mod_list": mod_list,
            "snr_list": snr_list
        },
        resume=True if checkpoint else False
    )
    
    # Get the SNR mapping from dataset
    snr_index_to_value = {idx: snr for snr, idx in train_loader.dataset.dataset.snr_labels.items()}
    
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
            print("Loaded model weights only (old format checkpoint)")
        print("Checkpoint loaded successfully.\n")
    else:
        print("\nNo checkpoint provided or checkpoint file not found, starting training from scratch.\n")
    
    # Training loop - use start_epoch
    for epoch in range(start_epoch, epochs):
        print(f"\nStarting Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0.0
        total_mod_loss = 0.0
        total_snr_loss = 0.0
        correct_mod = 0
        total_samples = 0
        snr_mae_sum = 0.0
        snr_acc_count = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, mod_target, snr_target) in enumerate(pbar):
            data, mod_target, snr_target = data.to(device), mod_target.to(device), snr_target.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training only for CUDA
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    mod_output, snr_output = model(data)
                    
                    # Calculate individual losses
                    mod_loss = criterion_modulation(mod_output, mod_target)
                    snr_loss = criterion_snr(snr_output, snr_target)
                    
                    # Combine losses using dynamic balancing
                    loss, weights = criterion_dynamic([mod_loss, snr_loss])
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass without mixed precision
                mod_output, snr_output = model(data)
                
                # Calculate individual losses
                mod_loss = criterion_modulation(mod_output, mod_target)
                snr_loss = criterion_snr(snr_output, snr_target)
                
                # Combine losses using dynamic balancing
                loss, weights = criterion_dynamic([mod_loss, snr_loss])
                
                # Normal backward pass
                loss.backward()
                optimizer.step()
            
            mod_weight, snr_weight = weights
            
            # Update metrics
            total_loss += loss.item()
            total_mod_loss += mod_loss.item()
            total_snr_loss += snr_loss.item()
            
            # Calculate modulation accuracy
            pred_mod = mod_output.argmax(dim=1)
            correct_mod += (pred_mod == mod_target).sum().item()
            total_samples += mod_target.size(0)
            
            # Calculate SNR metrics (classification)
            pred_snr_class = torch.argmax(snr_output, dim=1)
            snr_acc_count += (pred_snr_class == snr_target).sum().item()
            
            # For backward compatibility, calculate MAE using expected SNR values
            # Get expected SNR values using the same method as in WeightedSNRLoss
            pred_snr_values = criterion_snr.scale_to_snr(snr_output).squeeze()
            true_snr_values = torch.tensor([snr_index_to_value[idx.item()] for idx in snr_target], 
                                         device=device, dtype=torch.float32)
            snr_mae_sum += torch.abs(pred_snr_values - true_snr_values).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_mod_loss = total_mod_loss / (batch_idx + 1)
            avg_snr_loss = total_snr_loss / (batch_idx + 1)
            mod_acc = 100. * correct_mod / total_samples
            snr_mae = snr_mae_sum / total_samples
            snr_acc = 100 * snr_acc_count / total_samples
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.3f}',
                'Mod Loss': f'{avg_mod_loss:.3f}',
                'SNR Loss': f'{avg_snr_loss:.3f}',
                'Mod Acc': f'{mod_acc:.2f}%',
                'SNR MAE': f'{snr_mae:.2f} dB',
                'SNR Acc': f'{snr_acc:.2f}%',
                'Mod W': f'{mod_weight:.2f}',
                'SNR W': f'{snr_weight:.2f}'
            })
        
        # Calculate average metrics for the epoch
        avg_loss = total_loss / len(train_loader)
        modulation_accuracy = 100 * correct_mod / total_samples
        snr_mae = snr_mae_sum / total_samples
        snr_accuracy = 100 * snr_acc_count / total_samples
        avg_modulation_loss = total_mod_loss / len(train_loader)
        avg_snr_loss = total_snr_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct_modulation = 0
        val_total_modulation = 0
        val_snr_mae = 0
        val_snr_correct = 0
        val_total_snr = 0
        val_modulation_losses = []
        val_snr_losses = []
        
        with torch.no_grad():
            for inputs, modulation_targets, snr_targets in val_loader:
                inputs = inputs.to(device)
                modulation_targets = modulation_targets.to(device)
                snr_targets = snr_targets.to(device)
                
                modulation_output, snr_output = model(inputs)
                
                # Calculate individual losses
                loss_modulation = criterion_modulation(modulation_output, modulation_targets)
                loss_snr = criterion_snr(snr_output, snr_targets)
                
                # Combine losses using dynamic loss balancing
                total_val_loss, _ = criterion_dynamic([loss_modulation, loss_snr])
                val_loss += total_val_loss.item()
                
                # Track individual losses
                val_modulation_losses.append(loss_modulation.item())
                val_snr_losses.append(loss_snr.item())
                
                # Calculate modulation accuracy
                _, predicted_modulation = torch.max(modulation_output.data, 1)
                val_total_modulation += modulation_targets.size(0)
                val_correct_modulation += (predicted_modulation == modulation_targets).sum().item()
                
                # Calculate SNR classification accuracy
                pred_snr_class = torch.argmax(snr_output, dim=1)
                val_snr_correct += (pred_snr_class == snr_targets).sum().item()
                
                # For backward compatibility, calculate MAE using expected SNR values
                # Get expected SNR values using the same method as in WeightedSNRLoss
                pred_snr_values = criterion_snr.scale_to_snr(snr_output).squeeze()
                true_snr_values = torch.tensor([snr_index_to_value[idx.item()] for idx in snr_targets],
                                              device=device, dtype=torch.float32)
                val_snr_mae += torch.abs(pred_snr_values - true_snr_values).sum().item()
                
                val_total_snr += snr_targets.size(0)
        
        # Calculate average validation metrics
        val_loss = val_loss / len(val_loader)
        val_modulation_accuracy = 100 * val_correct_modulation / val_total_modulation
        val_snr_mae = val_snr_mae / val_total_snr
        val_snr_accuracy = 100 * val_snr_correct / val_total_snr
        val_avg_modulation_loss = sum(val_modulation_losses) / len(val_modulation_losses)
        val_avg_snr_loss = sum(val_snr_losses) / len(val_snr_losses)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training - Loss: {avg_loss:.4f}, Mod Acc: {modulation_accuracy:.2f}%, SNR MAE: {snr_mae:.2f} dB, SNR Acc: {snr_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Mod Acc: {val_modulation_accuracy:.2f}%, SNR MAE: {val_snr_mae:.2f} dB, SNR Acc: {val_snr_accuracy:.2f}%")
        print(f"Training Losses - Mod: {avg_modulation_loss:.4f}, SNR: {avg_snr_loss:.4f}")
        print(f"Validation Losses - Mod: {val_avg_modulation_loss:.4f}, SNR: {val_avg_snr_loss:.4f}")
        print(f"Task Weights - Modulation: {mod_weight:.4f}, SNR: {snr_weight:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        print("Logging metrics to wandb...")
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_modulation_accuracy": modulation_accuracy,
            "train_snr_mae": snr_mae,
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
        })
        
        print("Checking for best model...")
        # Save best model with additional information
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
            torch.save(checkpoint_data, model_path, _use_new_zipfile_serialization=True)
            print(f"Saved new best model to {model_path} with validation loss: {best_val_loss:.4f}")
        
        print("Updating learning rate scheduler...")
        # Update learning rate scheduler with validation loss
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
                    true_snr_values = torch.tensor([snr_index_to_value[idx.item()] for idx in snr_targets],
                                                  device=device, dtype=torch.float32)
                    
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
        # Plot and save confusion matrices
        modulation_class_names = list(train_loader.dataset.dataset.modulation_labels.keys())
        plot_validation_confusion_matrices(
            all_true_modulation, 
            all_pred_modulation, 
            all_true_snr, 
            all_pred_snr,
            mod_classes=modulation_class_names, 
            save_dir=confusion_matrices_dir,
            epoch=epoch+1
        )
        print(f"Saved confusion matrices to {confusion_matrices_dir}")
    
    # Close wandb
    wandb.finish()
    
    return os.path.join(save_dir, 'best_model.pth')
