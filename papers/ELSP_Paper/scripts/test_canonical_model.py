#!/usr/bin/env python3
"""
Test script for evaluating the canonical model (run lmp0536i epoch 14) on test set.
This script implements Phase 3A of the ELSP paper TODO.

Usage:
    uv run python papers/ELSP_Paper/scripts/test_canonical_model.py [--checkpoint PATH]

Expected outputs:
    - results/performance_metrics/test_set_results.json
    - results/confusion_matrices/modulation_confusion_matrix.png
    - results/confusion_matrices/snr_confusion_matrix.png
    - results/f1_scores/modulation_f1_scores.json
    - results/f1_scores/snr_f1_scores.json
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import wandb
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Import training modules
from loaders.constellation_loader import ConstellationDataset
from models.constellation_model import ConstellationResNet
from models.vision_transformer_model import ConstellationVisionTransformer
from models.swin_transformer_model import ConstellationSwinTransformer
from utils.data_splits import create_stratified_split
from losses.uncertainty_weighted_loss import AnalyticalUncertaintyWeightedLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Configuration
CANONICAL_RUN_ID = "lmp0536i"
CANONICAL_EPOCH = 14
RESULTS_DIR = Path("papers/ELSP_Paper/results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset configuration (digital modulations only)
MODULATION_CLASSES = [
    '32PSK', '16APSK', '32QAM', '32APSK', 'OQPSK', '8ASK',
    'BPSK', '8PSK', '4ASK', '16PSK', '64APSK', '128QAM',
    '128APSK', '64QAM', 'QPSK', '256QAM', '16QAM', '8PAM'
]

SNR_LEVELS = list(range(0, 31, 2))  # 0 to 30 dB, step 2 (16 levels)
NUM_MODULATIONS = len(MODULATION_CLASSES)
NUM_SNR_LEVELS = len(SNR_LEVELS)

def setup_directories():
    """Create all necessary directories for results."""
    directories = [
        RESULTS_DIR / "checkpoints",
        RESULTS_DIR / "confusion_matrices", 
        RESULTS_DIR / "f1_scores",
        RESULTS_DIR / "performance_metrics",
        RESULTS_DIR / "figures",
        RESULTS_DIR / "raw_data"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_canonical_checkpoint(checkpoint_path=None):
    """Find and copy the canonical model checkpoint."""
    print(f"Looking for canonical checkpoint from run {CANONICAL_RUN_ID} epoch {CANONICAL_EPOCH}")
    
    # If checkpoint path provided as parameter, use it
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"Using provided checkpoint: {checkpoint_path}")
            
            # Copy to results directory
            target_path = CHECKPOINT_DIR / "canonical_model_epoch_14.pth"
            
            import shutil
            shutil.copy2(checkpoint_path, target_path)
            
            # Save metadata
            metadata = {
                "run_id": CANONICAL_RUN_ID,
                "epoch": CANONICAL_EPOCH,
                "source": "provided_path",
                "original_path": str(checkpoint_path),
                "copy_timestamp": datetime.now().isoformat(),
                "original_filename": checkpoint_path.name
            }
            
            with open(CHECKPOINT_DIR / "checkpoint_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Checkpoint copied to: {target_path}")
            return target_path
        else:
            raise FileNotFoundError(f"Provided checkpoint path does not exist: {checkpoint_path}")
    
    # Look for local checkpoint in default location
    local_checkpoint_name = f"best_model_resnet50_epoch_{CANONICAL_EPOCH}.pth"
    local_checkpoint_path = Path("checkpoints") / local_checkpoint_name
    
    if local_checkpoint_path.exists():
        print(f"Found local checkpoint: {local_checkpoint_path}")
        
        # Copy to results directory
        target_path = CHECKPOINT_DIR / "canonical_model_epoch_14.pth"
        
        import shutil
        shutil.copy2(local_checkpoint_path, target_path)
        
        # Save metadata
        metadata = {
            "run_id": CANONICAL_RUN_ID,
            "epoch": CANONICAL_EPOCH,
            "source": "local_checkpoint",
            "original_path": str(local_checkpoint_path),
            "copy_timestamp": datetime.now().isoformat(),
            "original_filename": local_checkpoint_name
        }
        
        with open(CHECKPOINT_DIR / "checkpoint_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Checkpoint copied to: {target_path}")
        return target_path
    
    # If local checkpoint not found, try W&B as fallback
    print("Local checkpoint not found, attempting W&B download...")
    try:
        api = wandb.Api()
        run = api.run(f"shamoons/modulation-explainability/{CANONICAL_RUN_ID}")
        
        # Try to find checkpoint file
        checkpoint_file = None
        for file in run.files():
            if f"epoch_{CANONICAL_EPOCH}" in file.name and file.name.endswith('.pth'):
                checkpoint_file = file
                break
        
        if checkpoint_file is None:
            print(f"Warning: No checkpoint found for epoch {CANONICAL_EPOCH}")
            print("Available files:")
            for file in run.files():
                if file.name.endswith('.pth'):
                    print(f"  - {file.name}")
            
            # Try to find the best model checkpoint
            for file in run.files():
                if "best_model" in file.name and file.name.endswith('.pth'):
                    checkpoint_file = file
                    print(f"Using best model checkpoint: {file.name}")
                    break
        
        if checkpoint_file is None:
            raise FileNotFoundError("No suitable checkpoint found in W&B run")
        
        # Download checkpoint
        checkpoint_path = CHECKPOINT_DIR / "canonical_model_epoch_14.pth"
        checkpoint_file.download(root=str(CHECKPOINT_DIR), replace=True)
        
        # Rename to canonical name if needed
        downloaded_path = CHECKPOINT_DIR / checkpoint_file.name
        if downloaded_path != checkpoint_path:
            downloaded_path.rename(checkpoint_path)
        
        # Save metadata
        metadata = {
            "run_id": CANONICAL_RUN_ID,
            "epoch": CANONICAL_EPOCH,
            "source": "wandb_download",
            "download_timestamp": datetime.now().isoformat(),
            "original_filename": checkpoint_file.name,
            "run_config": dict(run.config),
            "run_summary": dict(run.summary)
        }
        
        with open(CHECKPOINT_DIR / "checkpoint_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Checkpoint downloaded to: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        raise FileNotFoundError(f"Could not find checkpoint locally or in W&B. Local path tried: {local_checkpoint_path}")

def load_model_and_checkpoint(checkpoint_path):
    """Load the model architecture and checkpoint."""
    print("Loading model architecture and checkpoint...")
    
    # Load checkpoint to get configuration
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Build ResNet50 model (canonical run configuration)
    model = ConstellationResNet(
        num_modulation_classes=NUM_MODULATIONS,
        num_snr_classes=NUM_SNR_LEVELS,
        architecture="resnet50",
        snr_layer_config="bottleneck_128",
        pretrained=True,
        dropout_rate=0.5
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, checkpoint

def create_test_dataloader():
    """Create test dataloader for evaluation."""
    print("Creating test dataloader...")
    
    # Use the same configuration as training
    data_dir = "constellation_diagrams"
    batch_size = 512
    
    # Create dataset (exclude analog modulations, use 0-30 dB SNR range)
    excluded_modulations = ['AM-SSB-SC', 'AM-DSB-SC', 'AM-SSB-WC', 'AM-DSB-WC', 'FM', 'GMSK', 'OOK']
    
    # Create dataset
    dataset = ConstellationDataset(
        root_dir=data_dir,
        snr_list=list(range(0, 31, 2)),  # 0-30 dB, step 2 (16 levels)
        mods_to_process=None  # Load all, then filter
    )
    
    # Filter out analog modulations
    filtered_indices = []
    for i, (_, mod_label, snr_label) in enumerate(dataset):
        mod_name = list(dataset.modulation_labels.keys())[list(dataset.modulation_labels.values()).index(mod_label)]
        if mod_name not in excluded_modulations:
            filtered_indices.append(i)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Filtered samples (digital only): {len(filtered_indices)}")
    
    # Create stratified split
    train_indices, val_indices, test_indices = create_stratified_split(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )
    
    # Filter test indices to only include digital modulations
    test_indices = [idx for idx in test_indices if idx in filtered_indices]
    
    print(f"Test set size: {len(test_indices)} samples")
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader

def evaluate_model(model, test_loader):
    """Evaluate model on test set and collect predictions."""
    print("Evaluating model on test set...")
    
    all_mod_preds = []
    all_snr_preds = []
    all_mod_labels = []
    all_snr_labels = []
    all_mod_probs = []
    all_snr_probs = []
    
    total_samples = 0
    correct_combined = 0
    correct_modulation = 0
    correct_snr = 0
    
    with torch.no_grad():
        for batch_idx, (images, mod_labels, snr_labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            mod_labels = mod_labels.to(DEVICE)
            snr_labels = snr_labels.to(DEVICE)
            
            # Forward pass
            mod_output, snr_output = model(images)
            
            # Get predictions
            mod_probs = F.softmax(mod_output, dim=1)
            snr_probs = F.softmax(snr_output, dim=1)
            
            _, mod_preds = torch.max(mod_output, 1)
            _, snr_preds = torch.max(snr_output, 1)
            
            # Calculate accuracies
            batch_size = images.size(0)
            total_samples += batch_size
            
            correct_modulation += (mod_preds == mod_labels).sum().item()
            correct_snr += (snr_preds == snr_labels).sum().item()
            correct_combined += ((mod_preds == mod_labels) & (snr_preds == snr_labels)).sum().item()
            
            # Store predictions and labels
            all_mod_preds.extend(mod_preds.cpu().numpy())
            all_snr_preds.extend(snr_preds.cpu().numpy())
            all_mod_labels.extend(mod_labels.cpu().numpy())
            all_snr_labels.extend(snr_labels.cpu().numpy())
            all_mod_probs.extend(mod_probs.cpu().numpy())
            all_snr_probs.extend(snr_probs.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx * batch_size}/{len(test_loader.dataset)} samples")
    
    # Calculate final accuracies
    modulation_accuracy = correct_modulation / total_samples
    snr_accuracy = correct_snr / total_samples
    combined_accuracy = correct_combined / total_samples
    
    print(f"Test Results:")
    print(f"  Combined Accuracy: {combined_accuracy:.4f} ({combined_accuracy*100:.2f}%)")
    print(f"  Modulation Accuracy: {modulation_accuracy:.4f} ({modulation_accuracy*100:.2f}%)")
    print(f"  SNR Accuracy: {snr_accuracy:.4f} ({snr_accuracy*100:.2f}%)")
    
    return {
        'predictions': {
            'modulation': all_mod_preds,
            'snr': all_snr_preds,
            'modulation_probs': all_mod_probs,
            'snr_probs': all_snr_probs
        },
        'labels': {
            'modulation': all_mod_labels,
            'snr': all_snr_labels
        },
        'accuracies': {
            'combined': combined_accuracy,
            'modulation': modulation_accuracy,
            'snr': snr_accuracy
        },
        'total_samples': total_samples
    }

def generate_confusion_matrices(results):
    """Generate and save confusion matrices."""
    print("Generating confusion matrices...")
    
    mod_preds = results['predictions']['modulation']
    snr_preds = results['predictions']['snr']
    mod_labels = results['labels']['modulation']
    snr_labels = results['labels']['snr']
    
    # Modulation confusion matrix
    mod_cm = confusion_matrix(mod_labels, mod_preds)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(mod_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=MODULATION_CLASSES, yticklabels=MODULATION_CLASSES)
    plt.title('Modulation Classification Confusion Matrix')
    plt.xlabel('Predicted Modulation')
    plt.ylabel('True Modulation')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices/modulation_confusion_matrix.png", dpi=300)
    plt.close()
    
    # SNR confusion matrix
    snr_cm = confusion_matrix(snr_labels, snr_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(snr_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SNR_LEVELS, yticklabels=SNR_LEVELS)
    plt.title('SNR Classification Confusion Matrix')
    plt.xlabel('Predicted SNR (dB)')
    plt.ylabel('True SNR (dB)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices/snr_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Save confusion matrix data
    cm_data = {
        'modulation_confusion_matrix': mod_cm.tolist(),
        'snr_confusion_matrix': snr_cm.tolist(),
        'modulation_classes': MODULATION_CLASSES,
        'snr_levels': SNR_LEVELS
    }
    
    with open(RESULTS_DIR / "confusion_matrices/confusion_matrix_data.json", 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    print("Confusion matrices saved successfully")

def calculate_f1_scores(results):
    """Calculate and save F1 scores."""
    print("Calculating F1 scores...")
    
    mod_preds = results['predictions']['modulation']
    snr_preds = results['predictions']['snr']
    mod_labels = results['labels']['modulation']
    snr_labels = results['labels']['snr']
    
    # Modulation F1 scores
    mod_precision, mod_recall, mod_f1, _ = precision_recall_fscore_support(
        mod_labels, mod_preds, average=None, zero_division=0
    )
    
    # SNR F1 scores
    snr_precision, snr_recall, snr_f1, _ = precision_recall_fscore_support(
        snr_labels, snr_preds, average=None, zero_division=0
    )
    
    # Create detailed F1 analysis
    mod_f1_data = {
        'per_class_f1': {
            MODULATION_CLASSES[i]: float(mod_f1[i]) for i in range(len(MODULATION_CLASSES))
        },
        'per_class_precision': {
            MODULATION_CLASSES[i]: float(mod_precision[i]) for i in range(len(MODULATION_CLASSES))
        },
        'per_class_recall': {
            MODULATION_CLASSES[i]: float(mod_recall[i]) for i in range(len(MODULATION_CLASSES))
        },
        'macro_avg_f1': float(np.mean(mod_f1)),
        'weighted_avg_f1': float(np.average(mod_f1, weights=np.bincount(mod_labels)))
    }
    
    snr_f1_data = {
        'per_class_f1': {
            str(SNR_LEVELS[i]): float(snr_f1[i]) for i in range(len(SNR_LEVELS))
        },
        'per_class_precision': {
            str(SNR_LEVELS[i]): float(snr_precision[i]) for i in range(len(SNR_LEVELS))
        },
        'per_class_recall': {
            str(SNR_LEVELS[i]): float(snr_recall[i]) for i in range(len(SNR_LEVELS))
        },
        'macro_avg_f1': float(np.mean(snr_f1)),
        'weighted_avg_f1': float(np.average(snr_f1, weights=np.bincount(snr_labels)))
    }
    
    # Save F1 scores
    with open(RESULTS_DIR / "f1_scores/modulation_f1_scores.json", 'w') as f:
        json.dump(mod_f1_data, f, indent=2)
    
    with open(RESULTS_DIR / "f1_scores/snr_f1_scores.json", 'w') as f:
        json.dump(snr_f1_data, f, indent=2)
    
    print("F1 scores calculated and saved successfully")
    return mod_f1_data, snr_f1_data

def save_test_results(results, mod_f1_data, snr_f1_data):
    """Save comprehensive test results."""
    print("Saving test results...")
    
    # Create comprehensive results summary
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'run_id': CANONICAL_RUN_ID,
        'epoch': CANONICAL_EPOCH,
        'model_config': {
            'architecture': 'resnet50',
            'snr_layer_type': 'bottleneck_128',
            'num_modulation_classes': NUM_MODULATIONS,
            'num_snr_classes': NUM_SNR_LEVELS
        },
        'test_set_size': results['total_samples'],
        'accuracies': results['accuracies'],
        'f1_scores': {
            'modulation': mod_f1_data,
            'snr': snr_f1_data
        },
        'detailed_metrics': {
            'modulation_classification_report': classification_report(
                results['labels']['modulation'], 
                results['predictions']['modulation'], 
                target_names=MODULATION_CLASSES,
                output_dict=True,
                zero_division=0
            ),
            'snr_classification_report': classification_report(
                results['labels']['snr'], 
                results['predictions']['snr'], 
                target_names=[str(snr) for snr in SNR_LEVELS],
                output_dict=True,
                zero_division=0
            )
        }
    }
    
    # Save main results
    with open(RESULTS_DIR / "performance_metrics/test_set_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save raw predictions and labels for further analysis
    np.save(RESULTS_DIR / "raw_data/test_predictions_modulation.npy", results['predictions']['modulation'])
    np.save(RESULTS_DIR / "raw_data/test_predictions_snr.npy", results['predictions']['snr'])
    np.save(RESULTS_DIR / "raw_data/test_labels_modulation.npy", results['labels']['modulation'])
    np.save(RESULTS_DIR / "raw_data/test_labels_snr.npy", results['labels']['snr'])
    np.save(RESULTS_DIR / "raw_data/test_probabilities_modulation.npy", results['predictions']['modulation_probs'])
    np.save(RESULTS_DIR / "raw_data/test_probabilities_snr.npy", results['predictions']['snr_probs'])
    
    print("Test results saved successfully")
    print(f"Main results file: {RESULTS_DIR / 'performance_metrics/test_set_results.json'}")

def main():
    """Main function to run the canonical model test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test canonical model on test set")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint file (default: checkpoints/best_model_resnet50_epoch_14.pth)"
    )
    
    args = parser.parse_args()
    
    print("Starting canonical model test evaluation...")
    print(f"Target run: {CANONICAL_RUN_ID} epoch {CANONICAL_EPOCH}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Setup
    setup_directories()
    
    # Get checkpoint
    checkpoint_path = get_canonical_checkpoint(args.checkpoint)
    
    # Load model
    model, checkpoint = load_model_and_checkpoint(checkpoint_path)
    
    # Create test dataloader
    test_loader = create_test_dataloader()
    
    # Evaluate model
    results = evaluate_model(model, test_loader)
    
    # Generate confusion matrices
    generate_confusion_matrices(results)
    
    # Calculate F1 scores
    mod_f1_data, snr_f1_data = calculate_f1_scores(results)
    
    # Save comprehensive results
    save_test_results(results, mod_f1_data, snr_f1_data)
    
    print("\n" + "="*50)
    print("TEST EVALUATION COMPLETE")
    print("="*50)
    print(f"Combined Accuracy: {results['accuracies']['combined']:.4f} ({results['accuracies']['combined']*100:.2f}%)")
    print(f"Modulation Accuracy: {results['accuracies']['modulation']:.4f} ({results['accuracies']['modulation']*100:.2f}%)")
    print(f"SNR Accuracy: {results['accuracies']['snr']:.4f} ({results['accuracies']['snr']*100:.2f}%)")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()