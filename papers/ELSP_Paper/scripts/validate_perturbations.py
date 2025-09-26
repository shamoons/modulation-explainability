#!/usr/bin/env python3
"""
Validate existing perturbations to ensure they match our requirements.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_perturbation(original_path, perturbed_path):
    """Analyze the perturbation between original and perturbed images."""
    # Load images
    original = np.array(Image.open(original_path).convert('L'))
    perturbed = np.array(Image.open(perturbed_path).convert('L'))
    
    # Calculate differences
    diff = np.abs(original.astype(float) - perturbed.astype(float))
    pixels_changed = np.sum(diff > 0)
    total_pixels = original.size
    percent_changed = (pixels_changed / total_pixels) * 100
    
    # Analyze which pixels were changed
    changed_mask = diff > 0
    changed_original_values = original[changed_mask]
    
    # Calculate statistics
    non_zero_pixels = original[original > 0]
    
    results = {
        'percent_changed': percent_changed,
        'pixels_changed': pixels_changed,
        'total_pixels': total_pixels,
        'mean_original_intensity': np.mean(original),
        'mean_perturbed_intensity': np.mean(perturbed),
        'changed_pixels_mean_intensity': np.mean(changed_original_values) if len(changed_original_values) > 0 else 0,
        'original_non_zero_pixels': len(non_zero_pixels),
        'perturbed_non_zero_pixels': np.sum(perturbed > 0)
    }
    
    return results, original, perturbed, diff

def validate_perturbation_types():
    """Validate different perturbation types."""
    # Test paths
    base_dir = "/home/shamoon/modulation-explainability"
    original_dir = os.path.join(base_dir, "constellation_diagrams/BPSK/SNR_10")
    perturbed_dir = os.path.join(base_dir, "perturbed_constellations/BPSK/SNR_10")
    
    # Sample file
    sample_name = "grayscale_BPSK_SNR_10_sample_0.png"
    original_path = os.path.join(original_dir, sample_name)
    
    # Check if paths exist
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        return
    
    # Test different perturbation types
    perturbation_types = [
        ('top1_blackout', '1% brightest pixels'),
        ('top5_blackout', '5% brightest pixels'),
        ('bottom1_blackout', '1% dimmest non-zero pixels'),
        ('bottom5_blackout', '5% dimmest non-zero pixels'),
        ('random1_blackout', '1% random pixels'),
        ('random5_blackout', '5% random pixels')
    ]
    
    print("Perturbation Validation Results:")
    print("=" * 60)
    
    for pert_type, description in perturbation_types:
        perturbed_name = sample_name.replace('.png', f'_{pert_type}.png')
        perturbed_path = os.path.join(perturbed_dir, perturbed_name)
        
        if not os.path.exists(perturbed_path):
            print(f"\n{description}: FILE NOT FOUND")
            continue
            
        results, original, perturbed, diff = analyze_perturbation(original_path, perturbed_path)
        
        print(f"\n{description} ({pert_type}):")
        print(f"  Actual % changed: {results['percent_changed']:.2f}%")
        print(f"  Pixels changed: {results['pixels_changed']:,} / {results['total_pixels']:,}")
        print(f"  Mean intensity of changed pixels: {results['changed_pixels_mean_intensity']:.2f}")
        print(f"  Original non-zero pixels: {results['original_non_zero_pixels']:,}")
        print(f"  Perturbed non-zero pixels: {results['perturbed_non_zero_pixels']:,}")
        
        # Check if it's masking the right regions
        if 'top' in pert_type:
            # For top perturbations, changed pixels should have high intensity
            original_values = original[diff > 0]
            if len(original_values) > 0:
                percentile = np.percentile(original[original > 0], 100 - float(pert_type[3]))
                above_threshold = np.sum(original_values >= percentile) / len(original_values)
                print(f"  % of changed pixels above {100-float(pert_type[3])} percentile: {above_threshold*100:.1f}%")
                
    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(original, cmap='hot')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Load and display perturbations
    positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    for idx, (pert_type, description) in enumerate(perturbation_types[:6]):
        if idx < len(positions):
            row, col = positions[idx]
            perturbed_name = sample_name.replace('.png', f'_{pert_type}.png')
            perturbed_path = os.path.join(perturbed_dir, perturbed_name)
            
            if os.path.exists(perturbed_path):
                perturbed = np.array(Image.open(perturbed_path).convert('L'))
                axes[row, col].imshow(perturbed, cmap='hot')
                axes[row, col].set_title(description)
                axes[row, col].axis('off')
    
    # Hide unused subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('perturbation_validation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'perturbation_validation.png'")
    
    # Check for missing percentages
    print("\n" + "=" * 60)
    print("Missing perturbation percentages:")
    required_percentages = [1, 2, 3, 4, 5]
    existing_percentages = [1, 5, 10]
    missing = set(required_percentages) - set(existing_percentages)
    if missing:
        print(f"Need to generate: {sorted(missing)}% perturbations")
    else:
        print("All required percentages are available")

if __name__ == "__main__":
    validate_perturbation_types()