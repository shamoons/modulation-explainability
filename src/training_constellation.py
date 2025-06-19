# src/training_constellation.py

from sklearn.model_selection import train_test_split
import torch
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
    uncertainty_weighter=None
):
    """
    Train the model and save the best one based on validation loss.
    Log metrics after validation and plot confusion matrices and F1 scores.
    Uses discrete SNR prediction (26 classes from -20 to 30 dB in 2dB steps).
    """
    save_dir = 'checkpoints'

    # Get the number of training and validation samples
    # num_train_samples = len(train_loader.sampler)
    # num_val_samples = len(val_loader.sampler)

    # Initialize WandB project
    wandb.init(project="modulation-explainability", config={
        "epochs": epochs,
        "mod_list": mod_list,
        "snr_list": snr_list,
        # "num_train_samples": num_train_samples,
        # "num_val_samples": num_val_samples,
        "model": model.model_name,
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "patience": patience
    })

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Initialize best validation loss
    model.to(device)

    # Initialize GradScaler for mixed-precision training (only if CUDA available)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(epochs):
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, shuffle=True)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Disable pin_memory for MPS
        use_pin_memory = device.type == 'cuda'
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12, pin_memory=use_pin_memory, prefetch_factor=4)

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

        train_modulation_accuracy = 100.0 * correct_modulation / total
        train_snr_accuracy = 100.0 * correct_snr / total
        train_combined_accuracy = 100.0 * correct_both / total
        train_loss = running_loss / len(train_loader)

        # Get current uncertainty weights and uncertainties
        current_weights = uncertainty_weighter.get_uncertainties() if uncertainty_weighter else None
        weight_mod, weight_snr = task_weights[0].item(), task_weights[1].item() if uncertainty_weighter else (0.5, 1.0)
        
        print(f"Epoch [{epoch+1}/{epochs}] Training Results:")
        print(f"  Train Loss (mod/snr): {train_loss:.4g} ({loss_modulation:.4g}/{loss_snr:.4g})")
        print(f"  Task Weights (mod/snr): {weight_mod:.3f}/{weight_snr:.3f}")
        if current_weights is not None:
            print(f"  Uncertainties (mod/snr): {current_weights[0]:.3f}/{current_weights[1]:.3f}")
        print(f"  Modulation Accuracy: {train_modulation_accuracy:.2f}%")
        print(f"  SNR Accuracy: {train_snr_accuracy:.2f}%")
        print(f"  Combined Accuracy: {train_combined_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        # Use autocast also in validation to speed up inference
        val_results = validate(model, device, criterion_modulation, criterion_snr, val_loader, use_autocast=True)
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

        scheduler.step(val_loss)

        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4g}")

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
