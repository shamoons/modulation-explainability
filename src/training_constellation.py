# src/training_constellation.py

from sklearn.model_selection import train_test_split
import torch
import wandb
from utils.image_utils import plot_f1_scores, plot_confusion_matrix
from utils.config_utils import load_loss_config
from validate_constellation import validate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler  # Import autocast and GradScaler from torch.amp
from torch.utils.data import random_split


def train(
    model,
    device,
    criterion_modulation,
    criterion_snr,
    criterion_dynamic,
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
    patience=1
):
    """
    Train the model with dynamic weighted loss.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Split dataset into train and validation sets (outside the training loop)
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize wandb
    wandb.init(
        project="constellation-classification",
        config={
            "model": "ConstellationResNet",
            "batch_size": batch_size,
            "learning_rate": base_lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "mod_list": mod_list,
            "snr_list": snr_list,
            "patience": patience
        }
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_modulation = 0
        total = 0
        modulation_losses = []
        snr_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, modulation_labels, snr_labels in progress_bar:
            inputs = inputs.to(device)
            modulation_labels = modulation_labels.to(device)
            snr_labels = snr_labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            modulation_output, snr_output = model(inputs)
            
            # Compute individual losses
            loss_modulation = criterion_modulation(modulation_output, modulation_labels)
            loss_snr = criterion_snr(snr_output, snr_labels)
            
            # Store individual losses for monitoring
            modulation_losses.append(loss_modulation.item())
            snr_losses.append(loss_snr.item())
            
            # Compute total loss using dynamic weighting
            total_loss = criterion_dynamic([loss_modulation, loss_snr])
            
            # Backward pass
            total_loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Update progress bar
            _, predicted_modulation = modulation_output.max(1)
            total += modulation_labels.size(0)
            correct_modulation += predicted_modulation.eq(modulation_labels).sum().item()
            
            # Get current weights for display
            weights = criterion_dynamic.get_weights()
            
            # Update progress bar with current metrics
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.3f}",
                'Mod Acc': f"{100.0 * correct_modulation / total:.2f}%",
                'SNR MAE': f"{loss_snr.item():.2f} dB",
                'Mod W': f"{weights[0]:.2f}",
                'SNR W': f"{weights[1]:.2f}"
            })
            
            # Log metrics to wandb
            wandb.log({
                'train_loss': total_loss.item(),
                'train_modulation_loss': loss_modulation.item(),
                'train_snr_loss': loss_snr.item(),
                'train_modulation_accuracy': 100.0 * correct_modulation / total,
                'modulation_weight': weights[0],
                'snr_weight': weights[1]
            })
        
        # Compute average losses for this epoch
        avg_modulation_loss = sum(modulation_losses) / len(modulation_losses)
        avg_snr_loss = sum(snr_losses) / len(snr_losses)
        
        # Validation
        val_loss, val_modulation_loss, val_snr_loss, val_modulation_accuracy, val_snr_mae, \
        all_true_modulation_labels, all_pred_modulation_labels, \
        all_true_snr_labels, all_pred_snr_labels = validate(
            model, device, criterion_modulation, criterion_snr, criterion_dynamic, val_loader
        )
        
        # Log validation metrics
        wandb.log({
            'val_loss': val_loss,
            'val_modulation_loss': val_modulation_loss,
            'val_snr_loss': val_snr_loss,
            'val_modulation_accuracy': val_modulation_accuracy,
            'val_snr_mae': val_snr_mae,
            'epoch': epoch
        })
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training Loss: {total_loss.item():.4f}")
        print(f"Training Modulation Loss: {avg_modulation_loss:.4f}")
        print(f"Training SNR Loss: {avg_snr_loss:.4f}")
        print(f"Training Modulation Accuracy: {100.0 * correct_modulation / total:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Modulation Accuracy: {val_modulation_accuracy:.2f}%")
        print(f"Validation SNR MAE: {val_snr_mae:.2f} dB")
        print(f"Current Weights - Modulation: {weights[0]:.4f}, SNR: {weights[1]:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Close wandb
    wandb.finish()
    
    return best_model_path
