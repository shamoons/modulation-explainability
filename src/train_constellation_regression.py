"""
Training script for constellation-based modulation classification with SNR regression.
Uses Swin Transformer with regression for SNR prediction instead of classification.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from dataset_constellation import ConstellationDataset
from models.swin_transformer_model_regression import create_swin_regression_model
from multi_task_loss import MultiTaskUncertaintyLoss
from utils import save_checkpoint, calculate_combined_accuracy, export_analysis_data


def train_epoch(model, dataloader, optimizer, uncertainty_loss, device, epoch):
    """Train for one epoch with regression."""
    model.train()
    
    running_loss = 0.0
    running_mod_loss = 0.0
    running_snr_loss = 0.0
    
    correct_mod = 0
    total_samples = 0
    
    snr_errors = []  # For regression metrics
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (images, mod_labels, snr_labels) in enumerate(progress_bar):
        images = images.to(device)
        mod_labels = mod_labels.to(device)
        snr_labels = snr_labels.to(device)
        
        # Convert SNR labels to continuous values (0-15 → 0-30 dB)
        snr_values = snr_labels.float() * 2.0
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        mod_loss = F.cross_entropy(outputs['modulation'], mod_labels)
        snr_loss = F.smooth_l1_loss(outputs['snr'].squeeze(), snr_values)
        
        # Optional: Add bounds penalty to keep predictions in [0, 30]
        snr_pred = outputs['snr'].squeeze()
        bounds_penalty = torch.mean(torch.relu(snr_pred - 30) + torch.relu(-snr_pred))
        
        # Combine losses with uncertainty weighting
        losses = {
            'modulation': mod_loss,
            'snr': snr_loss,
            'bounds': bounds_penalty * 0.1
        }
        
        total_loss, task_losses, task_weights = uncertainty_loss(losses)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += total_loss.item()
        running_mod_loss += task_losses['modulation'].item()
        running_snr_loss += task_losses['snr'].item()
        
        # Modulation accuracy
        _, mod_preds = outputs['modulation'].max(1)
        correct_mod += mod_preds.eq(mod_labels).sum().item()
        
        # SNR regression metrics
        snr_errors.extend(torch.abs(snr_pred - snr_values).cpu().numpy())
        
        total_samples += images.size(0)
        
        # Update progress bar
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'mod_acc': 100. * correct_mod / total_samples,
                'snr_mae': np.mean(snr_errors) if snr_errors else 0
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    mod_accuracy = 100. * correct_mod / total_samples
    snr_mae = np.mean(snr_errors)
    snr_rmse = np.sqrt(np.mean(np.array(snr_errors)**2))
    
    # Calculate "accuracy" for SNR by rounding to nearest 2 dB
    snr_rounded_correct = 0
    for i in range(len(dataloader.dataset)):
        if i >= len(snr_errors):
            break
        # Check if prediction rounds to correct 2 dB increment
        error = snr_errors[i]
        if error <= 1.0:  # Within 1 dB counts as correct when rounding to 2 dB
            snr_rounded_correct += 1
    
    snr_accuracy = 100. * snr_rounded_correct / len(snr_errors)
    
    return {
        'loss': epoch_loss,
        'mod_loss': running_mod_loss / len(dataloader),
        'snr_loss': running_snr_loss / len(dataloader),
        'mod_accuracy': mod_accuracy,
        'snr_accuracy': snr_accuracy,
        'snr_mae': snr_mae,
        'snr_rmse': snr_rmse,
        'task_weights': task_weights.cpu().numpy()
    }


def validate(model, dataloader, uncertainty_loss, device, epoch=0, export_analysis=False):
    """Validate the model with regression metrics."""
    model.eval()
    
    running_loss = 0.0
    running_mod_loss = 0.0
    running_snr_loss = 0.0
    
    correct_mod = 0
    total_samples = 0
    
    all_mod_labels = []
    all_mod_preds = []
    all_snr_labels = []
    all_snr_preds = []
    all_snr_errors = []
    
    with torch.no_grad():
        for images, mod_labels, snr_labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            mod_labels = mod_labels.to(device)
            snr_labels = snr_labels.to(device)
            
            # Convert SNR labels to continuous values
            snr_values = snr_labels.float() * 2.0
            
            # Forward pass
            outputs = model(images)
            
            # Compute losses
            mod_loss = F.cross_entropy(outputs['modulation'], mod_labels)
            snr_loss = F.smooth_l1_loss(outputs['snr'].squeeze(), snr_values)
            
            snr_pred = outputs['snr'].squeeze()
            bounds_penalty = torch.mean(torch.relu(snr_pred - 30) + torch.relu(-snr_pred))
            
            losses = {
                'modulation': mod_loss,
                'snr': snr_loss,
                'bounds': bounds_penalty * 0.1
            }
            
            total_loss, task_losses, _ = uncertainty_loss(losses)
            
            # Update metrics
            running_loss += total_loss.item()
            running_mod_loss += task_losses['modulation'].item()
            running_snr_loss += task_losses['snr'].item()
            
            # Modulation predictions
            _, mod_preds = outputs['modulation'].max(1)
            correct_mod += mod_preds.eq(mod_labels).sum().item()
            
            # Store for analysis
            all_mod_labels.extend(mod_labels.cpu().numpy())
            all_mod_preds.extend(mod_preds.cpu().numpy())
            all_snr_labels.extend(snr_labels.cpu().numpy())
            all_snr_preds.extend(snr_pred.cpu().numpy())
            all_snr_errors.extend(torch.abs(snr_pred - snr_values).cpu().numpy())
            
            total_samples += images.size(0)
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader)
    mod_accuracy = 100. * correct_mod / total_samples
    snr_mae = np.mean(all_snr_errors)
    snr_rmse = np.sqrt(np.mean(np.array(all_snr_errors)**2))
    
    # Calculate SNR "accuracy" by rounding to nearest 2 dB
    all_snr_values = np.array(all_snr_labels) * 2.0  # Convert to dB
    all_snr_pred_rounded = np.round(np.array(all_snr_preds) / 2.0) * 2.0
    all_snr_pred_rounded = np.clip(all_snr_pred_rounded, 0, 30)
    snr_correct = np.sum(all_snr_pred_rounded == all_snr_values)
    snr_accuracy = 100. * snr_correct / len(all_snr_values)
    
    # Export analysis data if requested
    if export_analysis:
        export_regression_analysis(
            all_mod_labels, all_mod_preds,
            all_snr_labels, all_snr_preds,
            epoch
        )
    
    return {
        'loss': val_loss,
        'mod_loss': running_mod_loss / len(dataloader),
        'snr_loss': running_snr_loss / len(dataloader),
        'mod_accuracy': mod_accuracy,
        'snr_accuracy': snr_accuracy,
        'snr_mae': snr_mae,
        'snr_rmse': snr_rmse,
        'within_2db': np.mean(np.array(all_snr_errors) <= 2.0) * 100
    }


def export_regression_analysis(mod_labels, mod_preds, snr_labels, snr_preds, epoch):
    """Export analysis data for regression model."""
    # Convert SNR predictions to discrete classes by rounding
    snr_values = np.array(snr_labels) * 2.0  # True dB values
    snr_pred_values = np.array(snr_preds)  # Predicted dB values
    snr_pred_classes = np.round(snr_pred_values / 2.0).astype(int)
    snr_pred_classes = np.clip(snr_pred_classes, 0, 15)
    
    # Export modulation analysis (same as before)
    export_analysis_data(mod_labels, mod_preds, snr_labels, snr_pred_classes, epoch)
    
    # Also export regression-specific analysis
    regression_df = pd.DataFrame({
        'true_snr_db': snr_values,
        'pred_snr_db': snr_pred_values,
        'error_db': np.abs(snr_pred_values - snr_values),
        'true_snr_class': snr_labels,
        'pred_snr_class': snr_pred_classes
    })
    
    regression_df.to_csv(f'regression_analysis/snr_regression_epoch_{epoch}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Train constellation classification with SNR regression')
    parser.add_argument('--data_dir', type=str, default='constellation_diagrams',
                        help='Directory containing constellation diagram images')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--model_type', type=str, default='swin_tiny',
                        choices=['swin_tiny', 'swin_small', 'swin_base'],
                        help='Type of Swin model to use')
    parser.add_argument('--use_dilated_preprocessing', type=bool, default=True,
                        help='Whether to use dilated CNN preprocessing')
    parser.add_argument('--dropout_prob', type=float, default=0.3,
                        help='Dropout probability used throughout the model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--export_freq', type=int, default=5,
                        help='Export analysis data every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="modulation-recognition",
        config={
            "architecture": f"{args.model_type}_regression",
            "dataset": "constellation_diagrams",
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "use_dilated_preprocessing": args.use_dilated_preprocessing,
            "dropout_prob": args.dropout_prob,
            "snr_prediction": "regression",
            "description": "SNR regression with consistent dropout and dilated CNN preprocessing"
        }
    )
    
    # Create datasets
    train_dataset = ConstellationDataset(args.data_dir, split='train')
    val_dataset = ConstellationDataset(args.data_dir, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    model = create_swin_regression_model(
        model_variant=args.model_type,
        num_classes=17,
        pretrained=True,
        input_channels=1,
        dropout_prob=args.dropout_prob,
        use_dilated_preprocessing=args.use_dilated_preprocessing
    ).to(device)
    
    # Loss and optimizer
    uncertainty_loss = MultiTaskUncertaintyLoss(num_tasks=3).to(device)  # mod, snr, bounds
    
    # Combine model and loss parameters
    params = list(model.parameters()) + list(uncertainty_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3, verbose=True
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training loop
    early_stopping_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, uncertainty_loss, device, epoch+1)
        
        # Validate
        val_metrics = validate(
            model, val_loader, uncertainty_loss, device, 
            epoch+1, export_analysis=(epoch+1) % args.export_freq == 0
        )
        
        # Calculate combined accuracy
        combined_train = calculate_combined_accuracy(
            train_metrics['mod_accuracy'], train_metrics['snr_accuracy']
        )
        combined_val = calculate_combined_accuracy(
            val_metrics['mod_accuracy'], val_metrics['snr_accuracy']
        )
        
        # Print results
        print(f"Epoch [{epoch+1}/{args.epochs}] Training Results:")
        print(f"  Train Loss (mod/snr): {train_metrics['loss']:.2f} "
              f"({train_metrics['mod_loss']:.4f}/{train_metrics['snr_loss']:.3f})")
        print(f"  Task Weights (mod/snr/bounds): "
              f"{train_metrics['task_weights'][0]:.3f}/"
              f"{train_metrics['task_weights'][1]:.3f}/"
              f"{train_metrics['task_weights'][2]:.3f}")
        print(f"  Modulation Accuracy: {train_metrics['mod_accuracy']:.2f}%")
        print(f"  SNR Accuracy (rounded): {train_metrics['snr_accuracy']:.2f}%")
        print(f"  SNR MAE: {train_metrics['snr_mae']:.2f} dB")
        print(f"  Combined Accuracy: {combined_train:.2f}%")
        
        print(f"Validation Results:")
        print(f"  Validation Loss (mod/snr): {val_metrics['loss']:.3f} "
              f"({val_metrics['mod_loss']:.3f}/{val_metrics['snr_loss']:.3f})")
        print(f"  Modulation Accuracy: {val_metrics['mod_accuracy']:.2f}%")
        print(f"  SNR Accuracy (rounded): {val_metrics['snr_accuracy']:.2f}%")
        print(f"  SNR MAE: {val_metrics['snr_mae']:.2f} dB")
        print(f"  SNR RMSE: {val_metrics['snr_rmse']:.2f} dB")
        print(f"  Within 2 dB: {val_metrics['within_2db']:.1f}%")
        print(f"  Combined Accuracy: {combined_val:.2f}%")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_mod_accuracy': train_metrics['mod_accuracy'],
            'train_snr_accuracy': train_metrics['snr_accuracy'],
            'train_snr_mae': train_metrics['snr_mae'],
            'train_combined_accuracy': combined_train,
            'val_loss': val_metrics['loss'],
            'val_mod_accuracy': val_metrics['mod_accuracy'],
            'val_snr_accuracy': val_metrics['snr_accuracy'],
            'val_snr_mae': val_metrics['snr_mae'],
            'val_snr_rmse': val_metrics['snr_rmse'],
            'val_within_2db': val_metrics['within_2db'],
            'val_combined_accuracy': combined_val,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'mod_weight': train_metrics['task_weights'][0],
            'snr_weight': train_metrics['task_weights'][1],
            'bounds_weight': train_metrics['task_weights'][2]
        })
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint if best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = f'checkpoints/best_model_{args.model_type}_regression_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path, best_val_loss)
            print(f"Best model saved: {checkpoint_path} with validation loss: {best_val_loss:.3f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement")
            break
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('regression_analysis', exist_ok=True)
    main()