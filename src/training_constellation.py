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
import numpy as np


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
    
    # Get the SNR mapping from dataset
    snr_index_to_value = {idx: snr for snr, idx in dataset.snr_labels.items()}
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
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
            
            # Convert SNR target indices to actual SNR values
            snr_values = torch.tensor([snr_index_to_value[idx.item()] for idx in snr_target], 
                                     device=device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            mod_output, snr_output = model(data)
            
            # Calculate individual losses
            mod_loss = criterion_modulation(mod_output, mod_target)
            snr_loss = criterion_snr(snr_output, snr_values)
            
            # Combine losses using dynamic balancing
            loss, weights = criterion_dynamic([mod_loss, snr_loss])
            mod_weight, snr_weight = weights
            
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mod_loss += mod_loss.item()
            total_snr_loss += snr_loss.item()
            
            # Calculate modulation accuracy
            pred_mod = mod_output.argmax(dim=1)
            correct_mod += (pred_mod == mod_target).sum().item()
            total_samples += mod_target.size(0)
            
            # Calculate SNR metrics
            pred_snr = criterion_snr.scale_to_snr(snr_output)
            snr_mae_sum += torch.abs(pred_snr - snr_values.unsqueeze(1)).sum().item()
            snr_acc_count += torch.abs(pred_snr - snr_values.unsqueeze(1)).le(2.0).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_mod_loss = total_mod_loss / (batch_idx + 1)
            avg_snr_loss = total_snr_loss / (batch_idx + 1)
            mod_acc = 100. * correct_mod / total_samples
            snr_mae = snr_mae_sum / total_samples
            snr_acc = 100. * snr_acc_count / total_samples
            
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
                
                # Convert SNR target indices to actual SNR values
                snr_values = torch.tensor([snr_index_to_value[idx.item()] for idx in snr_targets], 
                                         device=device, dtype=torch.float32)
                
                modulation_output, snr_output = model(inputs)
                
                # Calculate individual losses
                loss_modulation = criterion_modulation(modulation_output, modulation_targets)
                loss_snr = criterion_snr(snr_output, snr_values)
                
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
                
                # Calculate SNR metrics
                # Convert bounded [0,1] predictions to SNR range [-20, 30]
                predicted_snr = criterion_snr.scale_to_snr(snr_output)
                
                # SNR MAE
                val_snr_mae += torch.abs(predicted_snr - snr_values.unsqueeze(1)).sum().item()
                
                # SNR accuracy (within ±2 dB)
                val_snr_correct += torch.abs(predicted_snr - snr_values.unsqueeze(1)).le(2.0).sum().item()
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
        print(f"Task Weights - Modulation: {mod_weight:.4f}, SNR: {snr_weight:.4f}\n")
        
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
                
                # Convert SNR target indices to actual SNR values
                snr_values = torch.tensor([snr_index_to_value[idx.item()] for idx in snr_targets], 
                                         device=device, dtype=torch.float32)
                
                modulation_output, snr_output = model(inputs)
                
                # Get predicted modulation
                _, predicted_modulation = torch.max(modulation_output.data, 1)
                all_pred_modulation.extend(predicted_modulation.cpu().numpy())
                all_true_modulation.extend(modulation_targets.cpu().numpy())
                
                # Get predicted SNR
                # Convert bounded [0,1] predictions to SNR range [-20, 30]
                predicted_snr = criterion_snr.scale_to_snr(snr_output)
                all_pred_snr.extend(predicted_snr.cpu().numpy())
                all_true_snr.extend(snr_values.cpu().numpy())
        
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
        
        # Round SNR values for better visualization
        true_snr_rounded = np.round(np.array(all_true_snr))
        pred_snr_rounded = np.round(np.array(all_pred_snr).squeeze())  # Add squeeze to make it 1D
        
        # Create unique sorted SNR values
        unique_snrs = np.sort(np.unique(np.concatenate([true_snr_rounded, pred_snr_rounded])))
        
        # Create a mapping from SNR values to indices
        snr_to_idx = {snr: i for i, snr in enumerate(unique_snrs)}
        
        # Map SNR values to indices
        true_snr_indices = np.array([snr_to_idx[snr] for snr in true_snr_rounded])
        pred_snr_indices = np.array([snr_to_idx[snr] for snr in pred_snr_rounded])
        
        # Create confusion matrix
        cm_snr = confusion_matrix(true_snr_indices, pred_snr_indices)
        
        # Plot heatmap
        sns.heatmap(cm_snr, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_snrs, yticklabels=unique_snrs)
        plt.title(f'SNR Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted SNR (dB)')
        plt.ylabel('True SNR (dB)')
        plt.savefig(os.path.join(results_dir, f'snr_cm_epoch_{epoch+1}.png'))
        plt.close()
        
        # Update learning rate
        scheduler.step()
    
    # Close wandb
    wandb.finish()
    
    return os.path.join(save_dir, 'best_model.pth')
