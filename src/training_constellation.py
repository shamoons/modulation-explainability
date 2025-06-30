# src/training_constellation.py

import torch
import numpy as np
from utils.data_splits import create_stratified_split, verify_stratification
import wandb
from utils.image_utils import plot_f1_scores, plot_confusion_matrix
from validate_constellation import validate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler  # Import autocast and GradScaler
import torch


def train(
    model,
    device,
    criterion_modulation,
    criterion_snr,
    optimizer,
    scheduler,
    dataset,
    batch_size=64,
    test_size=0.2,
    epochs=10,
    save_dir="checkpoints",
    mod_list=None,
    snr_list=None,
    base_lr=None,
    weight_decay=None,
    patience=1,
    uncertainty_weighter=None,
    model_type=None,
    dropout=0.2,
    max_lr=None,
    step_size_up=5,
    step_size_down=5,
    snr_alpha=0.5
):
    """
    Train the model and save the best one based on validation loss.
    Log metrics after validation and plot confusion matrices and F1 scores.
    Uses discrete SNR classification with distance-weighted penalty to prevent
    28 dB black hole while maintaining classification benefits.
    """
    save_dir = 'checkpoints'

    # Get the number of training and validation samples
    # num_train_samples = len(train_loader.sampler)
    # num_val_samples = len(val_loader.sampler)

    # Initialize WandB project
    wandb_config = {
        "epochs": epochs,
        "mod_list": mod_list,
        "snr_list": snr_list,
        "model": model.model_name,
        "model_type": model_type,
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "dropout": dropout,
        "batch_size": batch_size,
        "snr_alpha": snr_alpha,
        "description": f"DISTANCE-WEIGHTED SNR CLASSIFICATION - {model_type} with cyclic LR and inverse-square distance penalty (alpha={snr_alpha}) for SNR. Using classification (16 classes) instead of regression, with penalty that heavily weights distant SNR predictions using 1/d² law. Alpha={snr_alpha}: {'pure CE loss' if snr_alpha == 0 else 'weak penalty' if snr_alpha < 0.5 else 'moderate penalty' if snr_alpha <= 1.0 else 'strong penalty'}. This prevents 28 dB black hole while maintaining classification benefits. CyclicLR: base={base_lr if base_lr else '1e-5'}, max={max_lr if max_lr else 10*base_lr if base_lr else '1e-4'}, triangular2 mode. Bounded SNR 0-30dB, SNR-preserving constellation generation."
    }
    
    
    wandb.init(project="modulation-explainability", config=wandb_config)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Initialize best validation loss
    model.to(device)

    # Initialize GradScaler for mixed-precision training (only if CUDA available)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    # Create stratified train/val/test split (done once, not per epoch)
    # Use default 80/10/10 split from create_stratified_split function
    train_idx, val_idx, test_idx = create_stratified_split(dataset, random_state=42)
    
    # Verify stratification (optional - can be disabled for speed)
    verify_stratification(dataset, train_idx, val_idx, test_idx)
    
    # Calculate actual split ratios for WandB logging
    total_samples = len(dataset)
    actual_train_ratio = len(train_idx) / total_samples
    actual_val_ratio = len(val_idx) / total_samples
    actual_test_ratio = len(test_idx) / total_samples
    
    # Update WandB config with actual split ratios and sample counts
    wandb.config.update({
        "train_split": actual_train_ratio,
        "val_split": actual_val_ratio, 
        "test_split": actual_test_ratio,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "total_samples": total_samples
    })
    
    # Create samplers - val is fixed, train shuffles each epoch
    val_sampler = SubsetRandomSampler(val_idx)  # Validation doesn't need shuffling
    
    # Track best epoch for final test evaluation
    best_epoch = 0
    
    # Early stopping variables
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(epochs):
        if early_stop:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best model was from epoch {best_epoch} with validation loss: {best_val_loss:.4g}")
            break
        
        # Shuffle train indices each epoch for better generalization
        np.random.shuffle(train_idx)
        train_sampler = SubsetRandomSampler(train_idx)

        # Disable pin_memory for MPS
        use_pin_memory = device.type == 'cuda'
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)
        
        # Create CyclicLR scheduler on first epoch
        if epoch == 0:
            # Use the provided base_lr directly
            actual_base_lr = base_lr if base_lr else 1e-5  # Use actual base_lr for cyclic lower bound
            # Set max_lr if not provided
            if max_lr is None:
                max_lr = 10 * base_lr if base_lr else 1e-4  # Default to 10x base_lr if not specified
            
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=actual_base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up * len(train_loader),  # Convert epochs to iterations
                step_size_down=step_size_down * len(train_loader),
                mode='triangular2',  # Halves amplitude each cycle
                gamma=1.0,
                cycle_momentum=False  # Don't cycle momentum for Adam
            )
            print(f"Using CyclicLR scheduler: base_lr={actual_base_lr}, max_lr={max_lr}, mode=triangular2")
            print(f"Cycle length: {step_size_up + step_size_down} epochs ({(step_size_up + step_size_down) * len(train_loader)} iterations)")
            print(f"Note: LR will vary continuously throughout training (per-batch updates)")

        model.train()
        running_loss = 0.0
        correct_modulation = 0
        correct_snr = 0
        correct_both = 0  # For combined accuracy
        total = 0

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} - Learning Rate: {current_lr}")

        # Training loop with tqdm progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as progress:
            for inputs, modulation_labels, snr_labels in progress:
                inputs = inputs.to(device)
                modulation_labels = modulation_labels.to(device)
                snr_labels = snr_labels.to(device)

                optimizer.zero_grad()

                # Forward pass under autocast for mixed precision (only for CUDA)
                if device.type == 'cuda':
                    with autocast('cuda'):
                        modulation_output, snr_output = model(inputs)
                else:
                    # For non-CUDA devices, don't use autocast or no_grad
                    modulation_output, snr_output = model(inputs)

                # Compute loss for both outputs
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                # SNR classification with distance penalty
                loss_snr = criterion_snr(snr_output, snr_labels)

                # Use analytical uncertainty weighting for multi-task learning
                total_loss, task_weights = uncertainty_weighter([loss_modulation, loss_snr])

                # Backpropagation with scaled gradients (only for CUDA)
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    # Clip gradients for all parameters in optimizer
                    all_params = []
                    for param_group in optimizer.param_groups:
                        all_params.extend(param_group['params'])
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    # Clip gradients for all parameters in optimizer  
                    all_params = []
                    for param_group in optimizer.param_groups:
                        all_params.extend(param_group['params'])
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()

                running_loss += total_loss.item()

                _, predicted_modulation = modulation_output.max(1)
                _, predicted_snr = snr_output.max(1)

                total += modulation_labels.size(0)
                correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
                correct_snr += predicted_snr.eq(snr_labels).sum().item()

                # Calculate combined accuracy
                correct_both += ((predicted_modulation == modulation_labels) & (predicted_snr == snr_labels)).sum().item()

                # Update progress bar
                progress.set_postfix({
                    'Loss': f"{total_loss.item():.4g}",
                    'Mod Acc': f"{100.0 * correct_modulation / total:.2f}%",
                    'SNR Acc': f"{100.0 * correct_snr / total:.2f}%"
                })
                
                # Step the scheduler (CyclicLR needs to be called every batch)
                if scheduler is not None:
                    scheduler.step()

        train_modulation_accuracy = 100.0 * correct_modulation / total
        train_snr_accuracy = 100.0 * correct_snr / total
        train_combined_accuracy = 100.0 * correct_both / total
        train_loss = running_loss / len(train_loader)

        # Get current uncertainty weights and uncertainties
        current_weights = uncertainty_weighter.get_uncertainties() if uncertainty_weighter else None
        weight_mod, weight_snr = task_weights[0].item(), task_weights[1].item() if uncertainty_weighter else (0.5, 1.0)
        
        # Calculate actual Kendall weights for display
        if current_weights is not None:
            actual_weight_mod = 1.0 / (2.0 * current_weights[0]**2)
            actual_weight_snr = 1.0 / (2.0 * current_weights[1]**2)
        else:
            actual_weight_mod, actual_weight_snr = weight_mod, weight_snr
        
        print(f"Epoch [{epoch+1}/{epochs}] Training Results:")
        print(f"  Train Loss (mod/snr): {train_loss:.4g} ({loss_modulation:.4g}/{loss_snr:.4g})")
        print(f"  Actual Weights (mod/snr): {actual_weight_mod:.3f}/{actual_weight_snr:.3f}")
        print(f"  Relative Weights (mod/snr): {weight_mod:.1%}/{weight_snr:.1%}")
        if current_weights is not None:
            print(f"  Uncertainties (mod/snr): {current_weights[0]:.3f}/{current_weights[1]:.3f}")
        print(f"  Modulation Accuracy: {train_modulation_accuracy:.2f}%")
        print(f"  SNR Accuracy: {train_snr_accuracy:.2f}%")
        print(f"  Combined Accuracy: {train_combined_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        # Only use autocast for CUDA devices
        use_autocast_val = device.type == 'cuda'
        val_results = validate(model, device, val_loader, criterion_modulation, criterion_snr, uncertainty_weighter, use_autocast=use_autocast_val)
        (
            val_loss,
            modulation_loss_total,
            snr_loss_total,
            val_modulation_accuracy,
            val_snr_accuracy,
            val_combined_accuracy,
            all_true_modulation_labels,
            all_pred_modulation_labels,
            all_true_snr_labels,
            all_pred_snr_labels
        ) = val_results

        # CyclicLR is stepped every batch in the training loop, not after validation

        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0  # Reset counter
            # Include model name in checkpoint filename
            model_name = model.model_name if hasattr(model, 'model_name') else 'unknown'
            checkpoint_name = f"best_model_{model_name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, checkpoint_name))
            print(f"Best model saved: {checkpoint_name} with validation loss: {best_val_loss:.4g}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss. Patience: {epochs_no_improve}/{early_stopping_patience}")
            if epochs_no_improve >= early_stopping_patience:
                early_stop = True

        fig_confusion_matrix_modulation = plot_confusion_matrix(
            all_true_modulation_labels,
            all_pred_modulation_labels,
            'Modulation',
            epoch,
            label_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()]
        )

        fig_confusion_matrix_snr = plot_confusion_matrix(
            all_true_snr_labels,
            all_pred_snr_labels,
            'SNR',
            epoch,
            label_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()]
        )
        fig_f1_scores_modulation = plot_f1_scores(
            all_true_modulation_labels,
            all_pred_modulation_labels,
            label_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()],
            label_type='Modulation',
            epoch=epoch
        )
        fig_f1_scores_snr = plot_f1_scores(
            all_true_snr_labels,
            all_pred_snr_labels,
            label_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()],
            label_type='SNR',
            epoch=epoch
        )

        print("Validation Results:")
        print(f"  Validation Loss (mod/snr): {val_loss:.4g} ({modulation_loss_total:.4g}/{snr_loss_total:.4g})")
        print(f"  Modulation Accuracy: {val_modulation_accuracy:.2f}%")
        print(f"  SNR Accuracy: {val_snr_accuracy:.2f}%")
        print(f"  Combined Accuracy: {val_combined_accuracy:.2f}%")

        # Log other metrics to WandB
        log_dict = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_modulation_accuracy": train_modulation_accuracy,
            "train_snr_accuracy": train_snr_accuracy,
            "train_combined_accuracy": train_combined_accuracy,
            "val_loss": val_loss,
            "val_modulation_accuracy": val_modulation_accuracy,
            "val_snr_accuracy": val_snr_accuracy,
            "val_combined_accuracy": val_combined_accuracy,
            "confusion_matrix_modulation": wandb.Image(fig_confusion_matrix_modulation),
            "confusion_matrix_snr": wandb.Image(fig_confusion_matrix_snr),
            "f1_scores_modulation": wandb.Image(fig_f1_scores_modulation),
            "f1_scores_snr": wandb.Image(fig_f1_scores_snr)
        }
        
        # Add uncertainty weighting metrics if available
        if uncertainty_weighter is not None:
            log_dict.update({
                "task_weight_modulation": weight_mod,
                "task_weight_snr": weight_snr,
                "uncertainty_modulation": current_weights[0].item() if current_weights is not None else 0,
                "uncertainty_snr": current_weights[1].item() if current_weights is not None else 0
            })
        
        wandb.log(log_dict)
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    # Load best model
    model_name = model.model_name if hasattr(model, 'model_name') else 'unknown'
    best_model_path = os.path.join(save_dir, f'best_model_{model_name}_epoch_{best_epoch}.pth')
    # Fallback to old naming convention if new file doesn't exist
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(save_dir, f'best_model_epoch_{best_epoch}.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {os.path.basename(best_model_path)}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Using final model (best model checkpoint not found)")
    
    # Create test loader
    test_sampler = SubsetRandomSampler(test_idx)
    use_pin_memory = device.type == 'cuda'
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, 
                            num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)
    
    # Run validation on test set
    test_results = validate(
        model, device, test_loader, criterion_modulation, criterion_snr, uncertainty_weighter
    )
    
    # Unpack all 10 values returned by validate
    (
        test_loss,
        test_modulation_loss,
        test_snr_loss,
        test_modulation_accuracy,
        test_snr_accuracy,
        test_combined_accuracy,
        _,  # true modulation labels
        _,  # pred modulation labels
        _,  # true snr labels
        _   # pred snr labels
    ) = test_results
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"  Test Loss: {test_loss:.3f}")
    print(f"  Test Modulation Accuracy: {test_modulation_accuracy:.2f}%")
    print(f"  Test SNR Accuracy: {test_snr_accuracy:.2f}%")
    print(f"  Test Combined Accuracy: {test_combined_accuracy:.2f}%")
    
    # Log final test results to wandb
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_modulation_accuracy": test_modulation_accuracy,
        "final_test_snr_accuracy": test_snr_accuracy,
        "final_test_combined_accuracy": test_combined_accuracy,
        "best_epoch": best_epoch
    })
    
    print(f"\nTraining completed! Best model saved at epoch {best_epoch}")
    print(f"Final test accuracy: {test_combined_accuracy:.2f}%")
