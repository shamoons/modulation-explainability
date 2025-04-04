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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
    weight_decay=None
):
    """
    Train the model with dynamic weighted loss.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create results directory for confusion matrices
    results_dir = os.path.join(save_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split dataset into train and validation sets (outside the training loop)
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Add fixed seed for reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    # Initialize wandb
    wandb.init(
        project="constellation-classification",
        config={
            "architecture": model.__class__.__name__,
            "batch_size": batch_size,
            "learning_rate": base_lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "mod_list": mod_list,
            "snr_list": snr_list,
            "test_size": test_size
        }
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0  # Initialize as float
        correct_modulation = 0
        total_modulation = 0
        snr_mae = 0.0  # Initialize as float
        snr_correct = 0
        total_snr = 0
        modulation_losses = []
        snr_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (inputs, modulation_targets, snr_targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            modulation_targets = modulation_targets.to(device)
            snr_targets = snr_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                modulation_output, snr_output = model(inputs)
                
                # Calculate individual losses
                loss_modulation = criterion_modulation(modulation_output, modulation_targets)
                loss_snr = criterion_snr(snr_output, snr_targets)
                
                # Combine losses using dynamic loss balancing
                batch_loss = criterion_dynamic([loss_modulation, loss_snr])
                total_loss += batch_loss.item()  # Accumulate loss
                
                # Track individual losses
                modulation_losses.append(loss_modulation.item())
                snr_losses.append(loss_snr.item())

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate modulation accuracy
            _, predicted_modulation = torch.max(modulation_output.data, 1)
            total_modulation += modulation_targets.size(0)
            correct_modulation += (predicted_modulation == modulation_targets).sum().item()
            
            # Calculate SNR metrics
            snr_probs = F.softmax(snr_output, dim=1)
            snr_values = torch.tensor(list(dataset.snr_labels.keys()), device=device)
            predicted_snr = torch.sum(snr_probs * snr_values, dim=1)
            true_snr = snr_values[snr_targets]
            
            # SNR MAE
            snr_mae += torch.abs(predicted_snr - true_snr).sum().item()
            
            # SNR accuracy (within ±2 dB)
            snr_correct += (torch.abs(predicted_snr - true_snr) <= 2).sum().item()
            total_snr += snr_targets.size(0)

            # Get current weights for display
            weights = criterion_dynamic.get_weights()
            
            # Calculate average loss so far
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar with current metrics
            progress_bar.set_postfix({
                'Loss': f"{avg_loss:.3f}",
                'Mod Loss': f"{loss_modulation.item():.3f}",
                'SNR Loss': f"{loss_snr.item():.3f}",
                'Mod Acc': f"{100.0 * correct_modulation / total_modulation:.2f}%",
                'SNR MAE': f"{snr_mae / total_snr:.2f} dB",
                'SNR Acc': f"{100.0 * snr_correct / total_snr:.2f}%",
                'Mod W': f"{weights[0]:.2f}",
                'SNR W': f"{weights[1]:.2f}"
            })
            progress_bar.update(1)
        
        # Calculate average metrics for the epoch
        avg_loss = total_loss / len(train_loader)
        modulation_accuracy = 100 * correct_modulation / total_modulation
        snr_mae = snr_mae / total_snr
        snr_accuracy = 100 * snr_correct / total_snr
        avg_modulation_loss = sum(modulation_losses) / len(modulation_losses)
        avg_snr_loss = sum(snr_losses) / len(snr_losses)
        
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
                total_val_loss = criterion_dynamic([loss_modulation, loss_snr])
                val_loss += total_val_loss.item()
                
                # Track individual losses
                val_modulation_losses.append(loss_modulation.item())
                val_snr_losses.append(loss_snr.item())
                
                # Calculate modulation accuracy
                _, predicted_modulation = torch.max(modulation_output.data, 1)
                val_total_modulation += modulation_targets.size(0)
                val_correct_modulation += (predicted_modulation == modulation_targets).sum().item()
                
                # Calculate SNR metrics
                snr_probs = F.softmax(snr_output, dim=1)
                snr_values = torch.tensor(list(dataset.snr_labels.keys()), device=device)
                predicted_snr = torch.sum(snr_probs * snr_values, dim=1)
                true_snr = snr_values[snr_targets]
                
                # SNR MAE
                val_snr_mae += torch.abs(predicted_snr - true_snr).sum().item()
                
                # SNR accuracy (within ±2 dB)
                val_snr_correct += (torch.abs(predicted_snr - true_snr) <= 2).sum().item()
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
        print(f"Task Weights - Modulation: {weights[0]:.4f}, SNR: {weights[1]:.4f}")
        
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # Save confusion matrices after each epoch
        # Get predictions for validation set
        all_pred_modulation = []
        all_true_modulation = []
        all_pred_snr = []
        all_true_snr = []
        
        with torch.no_grad():
            for inputs, modulation_targets, snr_targets in val_loader:
                inputs = inputs.to(device)
                modulation_targets = modulation_targets.to(device)
                snr_targets = snr_targets.to(device)
                
                modulation_output, snr_output = model(inputs)
                
                # Get predicted modulation
                _, predicted_modulation = torch.max(modulation_output.data, 1)
                all_pred_modulation.extend(predicted_modulation.cpu().numpy())
                all_true_modulation.extend(modulation_targets.cpu().numpy())
                
                # Get predicted SNR
                snr_probs = F.softmax(snr_output, dim=1)
                snr_values = torch.tensor(list(dataset.snr_labels.keys()), device=device)
                predicted_snr = torch.sum(snr_probs * snr_values, dim=1)
                all_pred_snr.extend(predicted_snr.cpu().numpy())
                all_true_snr.extend(snr_values[snr_targets].cpu().numpy())
        
        # Plot and save modulation confusion matrix
        plt.figure(figsize=(10, 8))
        cm_mod = confusion_matrix(all_true_modulation, all_pred_modulation)
        sns.heatmap(cm_mod, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Modulation Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'modulation_cm_epoch_{epoch+1}.png'))
        plt.close()
        
        # Plot and save SNR confusion matrix
        plt.figure(figsize=(10, 8))
        # Convert SNR values to class indices for confusion matrix
        snr_values = torch.tensor(list(dataset.snr_labels.keys()), device=device)
        # Ensure all tensors are on the same device
        true_snr_tensor = torch.tensor(all_true_snr, device=device)
        pred_snr_tensor = torch.tensor(all_pred_snr, device=device)
        snr_class_indices = torch.searchsorted(snr_values, true_snr_tensor).cpu().numpy()
        pred_snr_class_indices = torch.searchsorted(snr_values, pred_snr_tensor).cpu().numpy()
        cm_snr = confusion_matrix(snr_class_indices, pred_snr_class_indices)
        sns.heatmap(cm_snr, annot=True, fmt='d', cmap='Blues')
        plt.title(f'SNR Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'snr_cm_epoch_{epoch+1}.png'))
        plt.close()
        
        # Update learning rate
        scheduler.step()
    
    # Close wandb
    wandb.finish()
    
    return os.path.join(save_dir, 'best_model.pth')
