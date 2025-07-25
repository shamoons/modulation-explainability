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
    cycles_per_training=5,
    warmup_epochs=0,
    warmup_start_factor=0.1,
    curriculum_scheduler=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
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
        "warmup_epochs": warmup_epochs,
        "warmup_start_factor": warmup_start_factor,
        "description": f"ENHANCED SNR BOTTLENECK + DISTANCE PENALTY - {model_type} with enhanced SNR head (64-dim bottleneck) and distance-weighted SNR loss. SNR head: features → Linear(512,64) → ReLU → Dropout → Linear(64,16). Distance penalty: alpha * (pred_class - true_class)^2 penalizes distant predictions. CyclicLR: base={base_lr if base_lr else '1e-6'}, max={max_lr if max_lr else '1e-4'}, triangular2 mode. Bounded SNR 0-30dB, SNR-preserving constellation generation. Testing architectural + loss function synergy."
    }
    
    
    wandb.init(project="modulation-explainability", config=wandb_config)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Initialize best validation loss
    model.to(device)

    # Initialize GradScaler for mixed-precision training (only if CUDA available)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    # Create stratified train/val/test split (done once, not per epoch)
    # Use provided train/val/test ratios
    train_idx, val_idx, test_idx = create_stratified_split(
        dataset, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=42
    )
    
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
    
    # Calculate cycle-aware patience
    if cycles_per_training > 0:
        # Use adaptive cycles based on total epochs
        cycle_length = epochs / cycles_per_training
        early_stopping_patience = max(int(cycle_length * 2), 10)  # At least 2 cycles, minimum 10
        print(f"Using cycle-aware patience: {early_stopping_patience} epochs ({cycle_length:.1f} epochs per cycle)")
        
        # Override step_size_up and step_size_down for adaptive cycles
        step_size_up = int(cycle_length / 2)
        step_size_down = int(cycle_length / 2)
        print(f"Adaptive cycle parameters: step_size_up={step_size_up}, step_size_down={step_size_down}")
    else:
        # Use provided patience and step_size values
        early_stopping_patience = patience
        print(f"Using manual patience: {early_stopping_patience} epochs")
    
    # Early stopping variables
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(epochs):
        if early_stop:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best model was from epoch {best_epoch} with validation loss: {best_val_loss:.4g}")
            break
        
        # Update curriculum scheduler if enabled
        if curriculum_scheduler is not None:
            curriculum_scheduler.update_epoch(epoch)
            # Print distribution stats
            curriculum_scheduler.print_distribution(train_idx, val_idx, dataset)
        
        # Create data loaders with curriculum sampling if enabled
        if curriculum_scheduler is not None:
            from utils.curriculum_learning import create_curriculum_sampler
            train_loader = create_curriculum_sampler(
                dataset, train_idx, curriculum_scheduler, batch_size, shuffle=True
            )
            # Val loader doesn't use curriculum weighting
            val_loader = create_curriculum_sampler(
                dataset, val_idx, None, batch_size, shuffle=False
            )
        else:
            # Standard sampling without curriculum
            np.random.shuffle(train_idx)
            train_sampler = SubsetRandomSampler(train_idx)
            use_pin_memory = device.type == 'cuda'
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)
        
        # Create scheduler on first epoch
        if epoch == 0:
            # Use the provided base_lr directly
            actual_base_lr = base_lr if base_lr else 1e-5  # Use actual base_lr for cyclic lower bound
            # Set max_lr if not provided
            if max_lr is None:
                max_lr = 10 * base_lr if base_lr else 1e-4  # Default to 10x base_lr if not specified
            
            # Create scheduler based on warmup configuration
            if warmup_epochs > 0:
                # Create warmup scheduler first, then chain with CyclicLR
                warmup_steps = warmup_epochs * len(train_loader)
                
                # Create warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                
                # Create CyclicLR scheduler for after warmup
                cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=actual_base_lr,
                    max_lr=max_lr,
                    step_size_up=step_size_up * len(train_loader),
                    step_size_down=step_size_down * len(train_loader),
                    mode='triangular2',
                    gamma=1.0,
                    cycle_momentum=False
                )
                
                # Chain the schedulers using SequentialLR
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cyclic_scheduler],
                    milestones=[warmup_steps]
                )
                
                print(f"Using LR Warmup + CyclicLR scheduler:")
                print(f"  - Warmup: {warmup_epochs} epochs (start_factor={warmup_start_factor})")
                print(f"  - After warmup: CyclicLR with base_lr={actual_base_lr}, max_lr={max_lr}")
            else:
                # No warmup, use CyclicLR directly
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=actual_base_lr,
                    max_lr=max_lr,
                    step_size_up=step_size_up * len(train_loader),
                    step_size_down=step_size_down * len(train_loader),
                    mode='triangular2',
                    gamma=1.0,
                    cycle_momentum=False
                )
                
                print(f"Using CyclicLR scheduler (no warmup): base_lr={actual_base_lr}, max_lr={max_lr}, mode=triangular2")
            
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
            for batch_idx, (inputs, modulation_labels, snr_labels) in enumerate(progress):
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
                
                # Standard classification for SNR - use argmax
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
                # Skip the first step to avoid the warning about stepping before optimizer.step()
                if scheduler is not None and not (epoch == 0 and batch_idx == 0):
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
            # Only trigger early stopping after warmup period
            if epoch >= warmup_epochs and epochs_no_improve >= early_stopping_patience:
                early_stop = True

        fig_confusion_matrix_modulation, cm_modulation = plot_confusion_matrix(
            all_true_modulation_labels,
            all_pred_modulation_labels,
            'Modulation',
            epoch,
            label_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()]
        )

        fig_confusion_matrix_snr, cm_snr = plot_confusion_matrix(
            all_true_snr_labels,
            all_pred_snr_labels,
            'SNR',
            epoch,
            label_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()]
        )
        fig_f1_scores_modulation, f1_modulation = plot_f1_scores(
            all_true_modulation_labels,
            all_pred_modulation_labels,
            label_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()],
            label_type='Modulation',
            epoch=epoch
        )
        fig_f1_scores_snr, f1_snr = plot_f1_scores(
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
        
        # Add F1 scores per class to W&B
        mod_labels = [label for label in val_loader.dataset.inverse_modulation_labels.values()]
        snr_labels = [str(label) for label in val_loader.dataset.inverse_snr_labels.values()]
        
        for i, label in enumerate(mod_labels):
            log_dict[f"val_f1_scores_modulation_per_class/{label}"] = f1_modulation[i]
        
        for i, label in enumerate(snr_labels):
            log_dict[f"val_f1_scores_snr_per_class/{label}"] = f1_snr[i]
        
        # Log confusion matrices as tables
        wandb.log({
            "confusion_matrix_modulation_table": wandb.Table(
                columns=["True"] + mod_labels,
                data=[[mod_labels[i]] + list(cm_modulation[i]) for i in range(len(mod_labels))]
            ),
            "confusion_matrix_snr_table": wandb.Table(
                columns=["True"] + snr_labels,
                data=[[snr_labels[i]] + list(cm_snr[i]) for i in range(len(snr_labels))]
            ),
            "f1_scores_modulation_table": wandb.Table(
                columns=["Class", "F1_Score"],
                data=[[mod_labels[i], f1_modulation[i]] for i in range(len(mod_labels))]
            ),
            "f1_scores_snr_table": wandb.Table(
                columns=["SNR", "F1_Score"],
                data=[[snr_labels[i], f1_snr[i]] for i in range(len(snr_labels))]
            )
        })
        
        # Add uncertainty weighting metrics if available
        if uncertainty_weighter is not None:
            log_dict.update({
                "task_weight_modulation": weight_mod,
                "task_weight_snr": weight_snr,
                "uncertainty_modulation": current_weights[0].item() if current_weights is not None else 0,
                "uncertainty_snr": current_weights[1].item() if current_weights is not None else 0
            })
        
        wandb.log(log_dict)
        
        # Log curriculum distribution stats if enabled
        if curriculum_scheduler is not None:
            # Log training distribution stats
            train_dist = curriculum_scheduler.get_distribution_stats(train_idx, dataset)
            for key, value in train_dist.items():
                wandb.log({f"curriculum/train_{key}": value})
    
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
