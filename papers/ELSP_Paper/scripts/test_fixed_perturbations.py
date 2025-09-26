#!/usr/bin/env python3
"""
Test the fixed perturbation generation on a small subset.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from perturb_constellations import create_perturbations

def test_perturbations():
    """Test perturbation generation on a single image."""
    
    # Load a test image
    test_image_path = "/home/shamoon/modulation-explainability/constellation_diagrams/BPSK/SNR_10/grayscale_BPSK_SNR_10_sample_0.png"
    image = Image.open(test_image_path).convert('L')
    image_array = np.array(image)
    
    # Test perturbations with fixed code
    percents = [1, 2, 3, 4, 5]
    perturbations = create_perturbations(image_array, percents, include_random=True, random_seed=42)
    
    # Analyze results
    print("Perturbation Analysis Results:")
    print("=" * 60)
    print(f"Original image shape: {image_array.shape}")
    print(f"Total pixels: {image_array.size:,}")
    print(f"Non-zero pixels: {np.sum(image_array > 0):,} ({np.sum(image_array > 0)/image_array.size*100:.2f}%)")
    print(f"Mean intensity: {np.mean(image_array):.2f}")
    print(f"Mean non-zero intensity: {np.mean(image_array[image_array > 0]):.2f}")
    
    print("\nPerturbation Results:")
    print("-" * 60)
    
    for percent in percents:
        for pert_type in ['top', 'bottom', 'random']:
            key = f'{pert_type}{percent}_blackout'
            if key in perturbations:
                perturbed = perturbations[key]
                diff = np.abs(image_array.astype(float) - perturbed.astype(float))
                pixels_changed = np.sum(diff > 0)
                percent_changed = (pixels_changed / image_array.size) * 100
                
                # Analyze changed pixels
                changed_mask = diff > 0
                if np.any(changed_mask):
                    changed_original_values = image_array[changed_mask]
                    mean_intensity = np.mean(changed_original_values)
                else:
                    mean_intensity = 0
                
                print(f"\n{key}:")
                print(f"  Target: {percent}% | Actual: {percent_changed:.2f}%")
                print(f"  Pixels changed: {pixels_changed:,}")
                print(f"  Mean intensity of changed pixels: {mean_intensity:.2f}")
                
                # For top/bottom, check if we're getting the right pixels
                if pert_type == 'top' and np.sum(image_array > 0) > 0:
                    non_zero = image_array[image_array > 0]
                    threshold = np.percentile(non_zero, 100 - percent)
                    correct_pixels = np.sum(changed_original_values >= threshold) / len(changed_original_values)
                    print(f"  % of changed pixels ≥ {100-percent} percentile: {correct_pixels*100:.1f}%")
                elif pert_type == 'bottom' and np.sum(image_array > 0) > 0:
                    non_zero = image_array[image_array > 0]
                    threshold = np.percentile(non_zero, percent)
                    correct_pixels = np.sum(changed_original_values <= threshold) / len(changed_original_values)
                    print(f"  % of changed pixels ≤ {percent} percentile: {correct_pixels*100:.1f}%")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    # Original in first position
    axes[0, 0].imshow(image_array, cmap='hot')
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')
    
    # Fill in perturbations
    positions = {
        'top1': (0, 1), 'top2': (0, 2), 'top3': (0, 3), 'top4': (0, 4), 'top5': (0, 5),
        'bottom1': (1, 0), 'bottom2': (1, 1), 'bottom3': (1, 2), 'bottom4': (1, 3), 'bottom5': (1, 4),
        'random1': (2, 0), 'random2': (2, 1), 'random3': (2, 2), 'random4': (2, 3), 'random5': (2, 4)
    }
    
    for key, (row, col) in positions.items():
        full_key = f'{key}_blackout'
        if full_key in perturbations:
            axes[row, col].imshow(perturbations[full_key], cmap='hot')
            axes[row, col].set_title(f'{key}', fontsize=10)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    axes[1, 5].axis('off')
    axes[2, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_fixed_perturbations.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'test_fixed_perturbations.png'")

if __name__ == "__main__":
    test_perturbations()