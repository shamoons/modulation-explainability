#!/usr/bin/env python3
"""
Evaluate perturbation impact on canonical model (Phase 3B).

This script:
1. Loads the canonical model checkpoint
2. Evaluates performance on original test set
3. Evaluates performance on each perturbation type
4. Calculates PIS scores
5. Generates visualizations and analysis

Usage:
    uv run python papers/ELSP_Paper/scripts/evaluate_perturbations.py
    
Expected outputs:
    - results/perturbation_analysis/pis_summary.json
    - results/perturbation_analysis/detailed_results.json
    - results/perturbation_analysis/perturbation_impact_chart.png
    - results/perturbation_analysis/accuracy_degradation_curves.png
    - results/perturbation_analysis/modulation_specific_analysis.json
    - results/perturbation_analysis/snr_specific_analysis.json
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
from tqdm import tqdm
import argparse
import gc

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Import required modules
from loaders.constellation_loader import ConstellationDataset
from loaders.perturbation_loader import PerturbationDataset
from models.constellation_model import ConstellationResNet
from utils.data_splits import create_stratified_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Configuration
RESULTS_DIR = Path("papers/ELSP_Paper/results")
PERTURBATION_DIR = "perturbed_constellations"
CONSTELLATION_DIR = "constellation_diagrams"
CHECKPOINT_PATH = "checkpoints/best_model_resnet50_epoch_14.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perturbation configurations
PERTURBATION_CONFIGS = [
    # High-intensity perturbations
    {'type': 'top1_blackout', 'percent': 1, 'description': '1% brightest pixels'},
    {'type': 'top2_blackout', 'percent': 2, 'description': '2% brightest pixels'},
    {'type': 'top3_blackout', 'percent': 3, 'description': '3% brightest pixels'},
    {'type': 'top4_blackout', 'percent': 4, 'description': '4% brightest pixels'},
    {'type': 'top5_blackout', 'percent': 5, 'description': '5% brightest pixels'},
    
    # Low-intensity perturbations
    {'type': 'bottom1_blackout', 'percent': 1, 'description': '1% dimmest non-zero pixels'},
    {'type': 'bottom2_blackout', 'percent': 2, 'description': '2% dimmest non-zero pixels'},
    {'type': 'bottom3_blackout', 'percent': 3, 'description': '3% dimmest non-zero pixels'},
    {'type': 'bottom4_blackout', 'percent': 4, 'description': '4% dimmest non-zero pixels'},
    {'type': 'bottom5_blackout', 'percent': 5, 'description': '5% dimmest non-zero pixels'},
    
    # Random perturbations (baseline)
    {'type': 'random1_blackout', 'percent': 1, 'description': '1% random pixels'},
    {'type': 'random2_blackout', 'percent': 2, 'description': '2% random pixels'},
    {'type': 'random3_blackout', 'percent': 3, 'description': '3% random pixels'},
    {'type': 'random4_blackout', 'percent': 4, 'description': '4% random pixels'},
    {'type': 'random5_blackout', 'percent': 5, 'description': '5% random pixels'},
]

# Dataset configuration (from training)
SNR_LEVELS = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
ANALOG_MODS = ['AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC', 'FM', 'GMSK', 'OOK']

def setup_directories():
    """Create all necessary directories for results."""
    directories = [
        RESULTS_DIR / "perturbation_analysis",
        RESULTS_DIR / "perturbation_analysis" / "example_perturbations",
        RESULTS_DIR / "perturbation_analysis" / "detailed_results"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

def load_model(checkpoint_path):
    """Load the canonical model."""
    print("Loading canonical model...")
    
    # First create a dummy dataset to get the correct number of classes
    all_mods = [d for d in os.listdir(CONSTELLATION_DIR) if os.path.isdir(os.path.join(CONSTELLATION_DIR, d))]
    mods_to_process = [mod for mod in all_mods if mod not in ANALOG_MODS]
    
    NUM_MODULATIONS = len(mods_to_process)
    NUM_SNR_LEVELS = len(SNR_LEVELS)
    
    print(f"Model configuration: {NUM_MODULATIONS} modulations × {NUM_SNR_LEVELS} SNR levels")
    
    # Build model with same configuration as training
    model = ConstellationResNet(
        num_classes=NUM_MODULATIONS,
        snr_classes=NUM_SNR_LEVELS,
        input_channels=1,
        dropout_prob=0.5,
        model_name="resnet50",
        snr_layer_config="bottleneck_128",
        use_pretrained=True
    )
    
    # Load checkpoint
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    
    print("Model loaded successfully")
    return model, mods_to_process

def create_dataloaders(mods_to_process):
    """Create original test dataloader and get test indices."""
    print("Creating original test dataloader...")
    
    # Create original dataset
    original_dataset = ConstellationDataset(
        root_dir=CONSTELLATION_DIR,
        image_type='grayscale',
        snr_list=SNR_LEVELS,
        mods_to_process=mods_to_process
    )
    
    # Create same split as training
    train_indices, val_indices, test_indices = create_stratified_split(
        original_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )
    
    print(f"Test set size: {len(test_indices)} samples")
    
    # Create original test loader
    original_loader = DataLoader(
        original_dataset,
        batch_size=256,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=4,
        pin_memory=True
    )
    
    return original_loader, test_indices, mods_to_process

def evaluate_model(model, data_loader, description=""):
    """Evaluate model on a dataset."""
    if description:
        print(f"Evaluating: {description}")
    
    all_mod_preds = []
    all_snr_preds = []
    all_mod_labels = []
    all_snr_labels = []
    
    total_samples = 0
    correct_combined = 0
    correct_modulation = 0
    correct_snr = 0
    
    with torch.no_grad():
        for images, mod_labels, snr_labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            mod_labels = mod_labels.to(DEVICE)
            snr_labels = snr_labels.to(DEVICE)
            
            # Forward pass
            mod_output, snr_output = model(images)
            
            # Get predictions
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
    
    # Calculate accuracies
    modulation_accuracy = correct_modulation / total_samples if total_samples > 0 else 0
    snr_accuracy = correct_snr / total_samples if total_samples > 0 else 0
    combined_accuracy = correct_combined / total_samples if total_samples > 0 else 0
    
    # Calculate per-class metrics
    mod_precision, mod_recall, mod_f1, mod_support = precision_recall_fscore_support(
        all_mod_labels, all_mod_preds, average=None, zero_division=0
    )
    
    snr_precision, snr_recall, snr_f1, snr_support = precision_recall_fscore_support(
        all_snr_labels, all_snr_preds, average=None, zero_division=0
    )
    
    return {
        'accuracies': {
            'combined': combined_accuracy,
            'modulation': modulation_accuracy,
            'snr': snr_accuracy
        },
        'predictions': {
            'modulation': all_mod_preds,
            'snr': all_snr_preds
        },
        'labels': {
            'modulation': all_mod_labels,
            'snr': all_snr_labels
        },
        'per_class_metrics': {
            'modulation': {
                'f1': mod_f1,
                'precision': mod_precision,
                'recall': mod_recall,
                'support': mod_support
            },
            'snr': {
                'f1': snr_f1,
                'precision': snr_precision,
                'recall': snr_recall,
                'support': snr_support
            }
        },
        'total_samples': total_samples
    }

def evaluate_perturbation(model, perturbation_config, test_indices, mods_to_process, perturbation_dir):
    """Evaluate model on a specific perturbation type."""
    perturbation_type = perturbation_config['type']
    
    # Create perturbation dataset
    perturbed_dataset = PerturbationDataset(
        root_dir=CONSTELLATION_DIR,
        perturbation_dir=perturbation_dir,
        perturbation_type=perturbation_type,
        image_type='grayscale',
        snr_list=SNR_LEVELS,
        mods_to_process=mods_to_process
    )
    
    # Create loader with same test indices
    perturbed_loader = DataLoader(
        perturbed_dataset,
        batch_size=256,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluate_model(model, perturbed_loader, perturbation_config['description'])
    
    return results

def calculate_pis(original_accuracy, perturbed_accuracy, perturbation_fraction):
    """
    Calculate Perturbation Impact Score.
    PIS = ΔA/f where:
    - ΔA = accuracy drop (original - perturbed)
    - f = fraction of pixels perturbed
    """
    accuracy_drop = original_accuracy - perturbed_accuracy
    # Avoid division by zero
    if perturbation_fraction == 0:
        return {
            'accuracy_drop': accuracy_drop,
            'pis': 0,
            'relative_drop_percent': 0
        }
    
    pis = accuracy_drop / (perturbation_fraction / 100)  # Convert percent to fraction
    
    # Avoid division by zero for relative drop
    if original_accuracy == 0:
        relative_drop_percent = 0
    else:
        relative_drop_percent = (accuracy_drop / original_accuracy) * 100
    
    return {
        'accuracy_drop': accuracy_drop,
        'pis': pis,
        'relative_drop_percent': relative_drop_percent
    }

def analyze_results(baseline_results, all_perturbation_results):
    """Analyze all results and calculate PIS scores."""
    print("\nAnalyzing results...")
    
    analysis = {
        'baseline': baseline_results['accuracies'],
        'perturbations': {},
        'pis_scores': {},
        'summary': {}
    }
    
    # Analyze each perturbation
    for config, results in all_perturbation_results:
        perturbation_type = config['type']
        percent = config['percent']
        
        # Store accuracies
        analysis['perturbations'][perturbation_type] = {
            'config': config,
            'accuracies': results['accuracies']
        }
        
        # Calculate PIS for each metric
        pis_combined = calculate_pis(
            baseline_results['accuracies']['combined'],
            results['accuracies']['combined'],
            percent
        )
        
        pis_modulation = calculate_pis(
            baseline_results['accuracies']['modulation'],
            results['accuracies']['modulation'],
            percent
        )
        
        pis_snr = calculate_pis(
            baseline_results['accuracies']['snr'],
            results['accuracies']['snr'],
            percent
        )
        
        analysis['pis_scores'][perturbation_type] = {
            'combined': pis_combined,
            'modulation': pis_modulation,
            'snr': pis_snr
        }
    
    # Calculate summary statistics
    # Extract key PIS values for the paper
    analysis['summary'] = {
        'top1_pis': analysis['pis_scores']['top1_blackout']['combined']['pis'],
        'top5_pis': analysis['pis_scores']['top5_blackout']['combined']['pis'],
        'bottom1_pis': analysis['pis_scores']['bottom1_blackout']['combined']['pis'],
        'bottom5_pis': analysis['pis_scores']['bottom5_blackout']['combined']['pis'],
        'random1_pis': analysis['pis_scores']['random1_blackout']['combined']['pis'],
        'random5_pis': analysis['pis_scores']['random5_blackout']['combined']['pis'],
        'top_vs_random_ratio': (
            analysis['pis_scores']['top5_blackout']['combined']['pis'] / 
            analysis['pis_scores']['random5_blackout']['combined']['pis']
            if analysis['pis_scores']['random5_blackout']['combined']['pis'] > 0 else float('inf')
        )
    }
    
    return analysis

def generate_visualizations(analysis, detailed_results):
    """Generate all visualization plots."""
    print("Generating visualizations...")
    
    # 1. PIS vs Perturbation Percentage
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract data for plotting
    percentages = [1, 2, 3, 4, 5]
    top_pis = [analysis['pis_scores'][f'top{p}_blackout']['combined']['pis'] for p in percentages]
    bottom_pis = [analysis['pis_scores'][f'bottom{p}_blackout']['combined']['pis'] for p in percentages]
    random_pis = [analysis['pis_scores'][f'random{p}_blackout']['combined']['pis'] for p in percentages]
    
    # Plot lines
    ax.plot(percentages, top_pis, 'ro-', linewidth=2, markersize=8, label='Top pixels (brightest)')
    ax.plot(percentages, bottom_pis, 'bo-', linewidth=2, markersize=8, label='Bottom pixels (dimmest)')
    ax.plot(percentages, random_pis, 'go-', linewidth=2, markersize=8, label='Random pixels (baseline)')
    
    ax.set_xlabel('Perturbation Percentage (%)', fontsize=14)
    ax.set_ylabel('PIS (Perturbation Impact Score)', fontsize=14)
    ax.set_title('Perturbation Impact Analysis', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "perturbation_analysis" / "perturbation_impact_chart.png", dpi=300)
    plt.close()
    
    # 2. Accuracy Degradation Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['combined', 'modulation', 'snr']
    titles = ['Combined Accuracy', 'Modulation Accuracy', 'SNR Accuracy']
    baseline_accs = [analysis['baseline'][m] for m in metrics]
    
    for idx, (metric, title, baseline) in enumerate(zip(metrics, titles, baseline_accs)):
        ax = axes[idx]
        
        # Extract accuracy values
        top_accs = [analysis['perturbations'][f'top{p}_blackout']['accuracies'][metric] for p in percentages]
        bottom_accs = [analysis['perturbations'][f'bottom{p}_blackout']['accuracies'][metric] for p in percentages]
        random_accs = [analysis['perturbations'][f'random{p}_blackout']['accuracies'][metric] for p in percentages]
        
        # Add baseline (0% perturbation)
        all_percentages = [0] + percentages
        top_accs = [baseline] + top_accs
        bottom_accs = [baseline] + bottom_accs
        random_accs = [baseline] + random_accs
        
        # Plot
        ax.plot(all_percentages, top_accs, 'ro-', linewidth=2, markersize=8, label='Top pixels')
        ax.plot(all_percentages, bottom_accs, 'bo-', linewidth=2, markersize=8, label='Bottom pixels')
        ax.plot(all_percentages, random_accs, 'go-', linewidth=2, markersize=8, label='Random pixels')
        
        ax.set_xlabel('Perturbation Percentage (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "perturbation_analysis" / "accuracy_degradation_curves.png", dpi=300)
    plt.close()
    
    # 3. PIS Comparison Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    perturbation_types = ['Top 1%', 'Top 5%', 'Bottom 1%', 'Bottom 5%', 'Random 1%', 'Random 5%']
    pis_values = [
        analysis['summary']['top1_pis'],
        analysis['summary']['top5_pis'],
        analysis['summary']['bottom1_pis'],
        analysis['summary']['bottom5_pis'],
        analysis['summary']['random1_pis'],
        analysis['summary']['random5_pis']
    ]
    
    colors = ['red', 'darkred', 'blue', 'darkblue', 'green', 'darkgreen']
    bars = ax.bar(perturbation_types, pis_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, pis_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('PIS Score', fontsize=14)
    ax.set_title('Perturbation Impact Score Comparison', fontsize=16)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "perturbation_analysis" / "pis_comparison_bar_chart.png", dpi=300)
    plt.close()
    
    print("Visualizations saved successfully")

def save_results(analysis, detailed_results):
    """Save all analysis results."""
    print("Saving results...")
    
    # 1. Save PIS summary (main results for paper)
    pis_summary = {
        'baseline_accuracy': analysis['baseline'],
        'perturbation_results': {},
        'key_findings': analysis['summary'],
        'timestamp': datetime.now().isoformat()
    }
    
    # Add detailed results for each perturbation
    for pert_type, data in analysis['perturbations'].items():
        pis_summary['perturbation_results'][pert_type] = {
            'accuracy': data['accuracies'],
            'pis': analysis['pis_scores'][pert_type],
            'description': data['config']['description']
        }
    
    with open(RESULTS_DIR / "perturbation_analysis" / "pis_summary.json", 'w') as f:
        json.dump(pis_summary, f, indent=2)
    
    # 2. Save detailed results
    detailed_output = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'checkpoint_path': str(CHECKPOINT_PATH),
            'perturbation_configs': PERTURBATION_CONFIGS,
            'snr_levels': SNR_LEVELS
        },
        'baseline_results': detailed_results['baseline'],
        'perturbation_results': {
            config['type']: results 
            for config, results in detailed_results['perturbations']
        },
        'analysis': analysis
    }
    
    with open(RESULTS_DIR / "perturbation_analysis" / "detailed_results.json", 'w') as f:
        json.dump(detailed_output, f, indent=2)
    
    # 3. Per-modulation analysis
    modulation_analysis = analyze_by_modulation(
        detailed_results['baseline'], 
        detailed_results['perturbations']
    )
    
    with open(RESULTS_DIR / "perturbation_analysis" / "modulation_specific_analysis.json", 'w') as f:
        json.dump(modulation_analysis, f, indent=2)
    
    # 4. Per-SNR analysis
    snr_analysis = analyze_by_snr(
        detailed_results['baseline'], 
        detailed_results['perturbations']
    )
    
    with open(RESULTS_DIR / "perturbation_analysis" / "snr_specific_analysis.json", 'w') as f:
        json.dump(snr_analysis, f, indent=2)
    
    print(f"Results saved to {RESULTS_DIR / 'perturbation_analysis'}")

def analyze_by_modulation(baseline_results, perturbation_results):
    """Analyze PIS scores by modulation type."""
    # Get modulation classes from the baseline results
    num_mods = len(baseline_results['per_class_metrics']['modulation']['f1'])
    
    analysis = {}
    
    # For each perturbation type
    for config, results in perturbation_results:
        if config['type'] == 'top5_blackout':  # Focus on top5 for modulation analysis
            mod_f1_baseline = baseline_results['per_class_metrics']['modulation']['f1']
            mod_f1_perturbed = results['per_class_metrics']['modulation']['f1']
            
            # Calculate per-modulation PIS
            for i in range(num_mods):
                f1_drop = mod_f1_baseline[i] - mod_f1_perturbed[i]
                pis = f1_drop / (config['percent'] / 100)
                
                # Store by modulation index (we don't have names here)
                analysis[f'mod_{i}'] = {
                    'baseline_f1': float(mod_f1_baseline[i]),
                    'perturbed_f1': float(mod_f1_perturbed[i]),
                    'f1_drop': float(f1_drop),
                    'pis': float(pis)
                }
    
    return analysis

def analyze_by_snr(baseline_results, perturbation_results):
    """Analyze PIS scores by SNR level."""
    analysis = {}
    
    # For each perturbation type
    for config, results in perturbation_results:
        if config['type'] == 'top5_blackout':  # Focus on top5 for SNR analysis
            snr_f1_baseline = baseline_results['per_class_metrics']['snr']['f1']
            snr_f1_perturbed = results['per_class_metrics']['snr']['f1']
            
            # Calculate per-SNR PIS
            for i, snr in enumerate(SNR_LEVELS):
                f1_drop = snr_f1_baseline[i] - snr_f1_perturbed[i]
                pis = f1_drop / (config['percent'] / 100)
                
                analysis[str(snr)] = {
                    'baseline_f1': float(snr_f1_baseline[i]),
                    'perturbed_f1': float(snr_f1_perturbed[i]),
                    'f1_drop': float(f1_drop),
                    'pis': float(pis)
                }
    
    return analysis

def generate_example_perturbations():
    """Save example perturbed images for the paper."""
    print("Generating example perturbations...")
    
    # Select a clear example: QPSK at 20 dB
    example_mod = 'QPSK'
    example_snr = 20
    example_sample = 0
    
    # Original image path
    original_path = Path(CONSTELLATION_DIR) / example_mod / f"SNR_{example_snr}" / f"grayscale_{example_mod}_SNR_{example_snr}_sample_{example_sample}.png"
    
    if not original_path.exists():
        print(f"Warning: Example image not found at {original_path}")
        return
    
    # Copy original
    from PIL import Image
    import shutil
    
    output_dir = RESULTS_DIR / "perturbation_analysis" / "example_perturbations"
    
    # Copy original
    shutil.copy2(original_path, output_dir / "original_qpsk_20db.png")
    
    # Copy perturbations
    for pert_type in ['top1', 'top5', 'bottom1', 'bottom5', 'random1', 'random5']:
        perturbed_path = Path(PERTURBATION_DIR) / example_mod / f"SNR_{example_snr}" / f"grayscale_{example_mod}_SNR_{example_snr}_sample_{example_sample}_{pert_type}_blackout.png"
        
        if perturbed_path.exists():
            shutil.copy2(perturbed_path, output_dir / f"{pert_type}_qpsk_20db.png")
        else:
            print(f"Warning: Perturbed example not found at {perturbed_path}")
    
    print("Example perturbations saved")

def main():
    """Main function to run perturbation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate perturbation impact on canonical model")
    parser.add_argument(
        "--checkpoint", 
        type=str,
        default=CHECKPOINT_PATH,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--perturbation-dir",
        type=str,
        default=PERTURBATION_DIR,
        help="Directory containing perturbed images"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Update paths from arguments
    checkpoint_path = args.checkpoint
    perturbation_dir = args.perturbation_dir
    
    print("Starting perturbation impact evaluation...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Perturbation directory: {perturbation_dir}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Device: {DEVICE}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup
    setup_directories()
    
    # Load model
    model, mods_to_process = load_model(checkpoint_path)
    
    # Create dataloaders
    original_loader, test_indices, _ = create_dataloaders(mods_to_process)
    
    # Evaluate baseline performance
    print("\n" + "="*50)
    print("BASELINE EVALUATION")
    print("="*50)
    baseline_results = evaluate_model(model, original_loader, "Original constellation diagrams")
    
    print(f"\nBaseline Results:")
    print(f"  Combined: {baseline_results['accuracies']['combined']:.4f} ({baseline_results['accuracies']['combined']*100:.2f}%)")
    print(f"  Modulation: {baseline_results['accuracies']['modulation']:.4f} ({baseline_results['accuracies']['modulation']*100:.2f}%)")
    print(f"  SNR: {baseline_results['accuracies']['snr']:.4f} ({baseline_results['accuracies']['snr']*100:.2f}%)")
    
    # Evaluate all perturbations
    print("\n" + "="*50)
    print("PERTURBATION EVALUATION")
    print("="*50)
    
    all_perturbation_results = []
    
    for config in PERTURBATION_CONFIGS:
        print(f"\nEvaluating: {config['description']}")
        try:
            results = evaluate_perturbation(model, config, test_indices, mods_to_process, perturbation_dir)
            all_perturbation_results.append((config, results))
            
            # Print results
            print(f"  Combined: {results['accuracies']['combined']:.4f} ({results['accuracies']['combined']*100:.2f}%)")
            print(f"  Modulation: {results['accuracies']['modulation']:.4f} ({results['accuracies']['modulation']*100:.2f}%)")
            print(f"  SNR: {results['accuracies']['snr']:.4f} ({results['accuracies']['snr']*100:.2f}%)")
            
            # Clear GPU cache to prevent memory issues
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"  Error evaluating {config['type']}: {e}")
            # Add dummy results to maintain order
            dummy_results = {
                'accuracies': {'combined': 0, 'modulation': 0, 'snr': 0},
                'predictions': {'modulation': [], 'snr': []},
                'labels': {'modulation': [], 'snr': []},
                'per_class_metrics': {
                    'modulation': {'f1': np.zeros(17), 'precision': np.zeros(17), 
                                  'recall': np.zeros(17), 'support': np.zeros(17)},
                    'snr': {'f1': np.zeros(16), 'precision': np.zeros(16), 
                           'recall': np.zeros(16), 'support': np.zeros(16)}
                },
                'total_samples': 0
            }
            all_perturbation_results.append((config, dummy_results))
    
    # Analyze results
    analysis = analyze_results(baseline_results, all_perturbation_results)
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    print(f"Top 1% PIS: {analysis['summary']['top1_pis']:.2f}")
    print(f"Top 5% PIS: {analysis['summary']['top5_pis']:.2f}")
    print(f"Bottom 1% PIS: {analysis['summary']['bottom1_pis']:.2f}")
    print(f"Bottom 5% PIS: {analysis['summary']['bottom5_pis']:.2f}")
    print(f"Random 1% PIS: {analysis['summary']['random1_pis']:.2f}")
    print(f"Random 5% PIS: {analysis['summary']['random5_pis']:.2f}")
    print(f"Top vs Random ratio: {analysis['summary']['top_vs_random_ratio']:.2f}x")
    
    # Generate visualizations
    generate_visualizations(analysis, {
        'baseline': baseline_results,
        'perturbations': all_perturbation_results
    })
    
    # Save all results
    save_results(analysis, {
        'baseline': baseline_results,
        'perturbations': all_perturbation_results
    })
    
    # Generate example perturbations
    generate_example_perturbations()
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Results saved to: {RESULTS_DIR / 'perturbation_analysis'}")
    
    # Print LaTeX-ready snippets for the paper
    print("\n" + "="*50)
    print("LATEX SNIPPETS FOR PAPER")
    print("="*50)
    print(f"% Line 312: Top 5% accuracy drop and PIS")
    top5_results = next((r for c, r in all_perturbation_results if c['type'] == 'top5_blackout'), None)
    if top5_results:
        mod_drop = (baseline_results['accuracies']['modulation'] - top5_results[1]['accuracies']['modulation']) * 100
        print(f"modulation accuracy dropping from {baseline_results['accuracies']['modulation']*100:.2f}\\% to {top5_results[1]['accuracies']['modulation']*100:.2f}\\%")
        print(f"PIS of {analysis['summary']['top5_pis']:.1f}")
    
    print(f"\n% Line 315: Bottom perturbation PIS")
    print(f"PIS as low as {analysis['summary']['bottom1_pis']:.1f}")
    
    print(f"\n% Line 435: Summary PIS values")
    print(f"PIS up to {analysis['summary']['top1_pis']:.1f}")
    print(f"PIS < {max(analysis['summary']['bottom1_pis'], analysis['summary']['bottom5_pis']):.1f}")

if __name__ == "__main__":
    main()