# src/training_constellation.py

import torch
import wandb
from utils.image_utils import plot_f1_scores, plot_confusion_matrix
from utils.config_utils import load_loss_config
from validate_constellation import validate
from tqdm import tqdm
import os


def train(
    model,
    device,
    criterion_modulation,
    criterion_snr,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    epochs=10,
    save_dir="checkpoints",
    mod_list=None,
    snr_list=None,
    use_snr_buckets=False,
    base_lr=None,
    max_lr=None,
    weight_decay=None,
    num_cycles=1
):
    """
    Train the model and save the best one based on validation loss.
    Log metrics after validation and plot confusion matrices and F1 scores.
    If use_snr_buckets is True, SNR values will be classified into buckets (low, medium, high).
    """
    alpha, beta = load_loss_config()
    save_dir = 'checkpoints'

    # Get the number of training and validation samples
    num_train_samples = len(train_loader.sampler)
    num_val_samples = len(val_loader.sampler)

    # Initialize WandB project
    wandb.init(project="modulation-explainability", config={
        "epochs": epochs,
        "mod_list": mod_list,
        "snr_list": snr_list,
        "use_snr_buckets": use_snr_buckets,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "alpha": alpha,
        "beta": beta,
        "model": model.model_name,
        "base_lr": base_lr,
        "max_lr": max_lr,
        "weight_decay": weight_decay,
        "num_cycles": num_cycles
    })

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Initialize best validation loss

    model.to(device)

    for epoch in range(epochs):
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

                # Forward pass
                modulation_output, snr_output = model(inputs)

                # Compute loss for both outputs
                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)

                # Combine both losses
                total_loss = alpha * loss_modulation + beta * loss_snr
                total_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Adjust learning rate using scheduler
                scheduler.step()

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

        print(f"Epoch [{epoch+1}/{epochs}] Training Results:")
        print(f"  Train Loss (mod/snr): {train_loss:.4g} ({loss_modulation:.4g}/{loss_snr:.4g})")
        print(f"  Modulation Accuracy: {train_modulation_accuracy:.2f}%")
        print(f"  SNR Accuracy: {train_snr_accuracy:.2f}%")
        print(f"  Combined Accuracy: {train_combined_accuracy:.2f}%")

        # Perform validation at the end of each epoch
        val_results = validate(model, device, criterion_modulation, criterion_snr, val_loader, use_snr_buckets=use_snr_buckets)
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

        print(f"Validation Results:")
        print(f"  Validation Loss (mod/snr): {val_loss:.4g} ({modulation_loss_total:.4g}/{snr_loss_total:.4g})")
        print(f"  Modulation Accuracy: {val_modulation_accuracy:.2f}%")
        print(f"  SNR Accuracy: {val_snr_accuracy:.2f}%")
        print(f"  Combined Accuracy: {val_combined_accuracy:.2f}%")

        # Log other metrics to WandB
        wandb.log({
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
        })

        # Log weights after each epoch
        # log_weights_to_wandb(model, epoch)
