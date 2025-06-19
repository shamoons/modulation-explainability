# src/perturb_constellations.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial


def parse_modulation_and_snr(filepath):
    """
    Parse the modulation type and SNR from the file path.
    Assumes folder structure: 'modulation/SNR/image.png'.
    """
    parts = filepath.split(os.sep)
    if len(parts) >= 3:
        modulation_type = parts[-3]  # Parent folder of the SNR folder
        snr = parts[-2].replace("SNR_", "")  # Remove "SNR_" prefix
    else:
        modulation_type = "Unknown"
        snr = "Unknown"
    return modulation_type, snr


def create_perturbations(image_array, percents, include_random=False, random_seed=42):
    """
    Create all perturbation variants for a single image efficiently.
    
    Args:
        image_array: numpy array of the image
        percents: list of percentages to perturb
        include_random: whether to include random perturbations
        random_seed: seed for reproducible random perturbations
    
    Returns:
        dict: {perturbation_name: perturbed_array}
    """
    perturbations = {}
    
    # Pre-compute thresholds for all percentages (more efficient)
    non_zero_pixels = image_array[image_array > 0]
    has_non_zero = non_zero_pixels.size > 0
    
    # Cache percentile calculations
    top_thresholds = {p: np.percentile(image_array, 100-p) for p in percents}
    bottom_thresholds = {p: np.percentile(non_zero_pixels, p) if has_non_zero else 0 
                        for p in percents}
    
    for percent in percents:
        # Top X% perturbation
        mask_top = image_array >= top_thresholds[percent]
        top_perturbed = image_array.copy()
        top_perturbed[mask_top] = 0
        perturbations[f'top{percent}_blackout'] = top_perturbed
        
        # Bottom X% perturbation  
        if has_non_zero:
            mask_bottom = (image_array > 0) & (image_array <= bottom_thresholds[percent])
            bottom_perturbed = image_array.copy()
            bottom_perturbed[mask_bottom] = 0
            perturbations[f'bottom{percent}_blackout'] = bottom_perturbed
        else:
            # Handle edge case: no non-zero pixels
            perturbations[f'bottom{percent}_blackout'] = image_array.copy()
        
        # Random X% perturbation (baseline for explainability analysis)
        # Purpose: If random perturbation has similar impact as targeted perturbation,
        # it suggests those regions aren't specifically critical for classification
        if include_random:
            np.random.seed(random_seed)  # Reproducible random perturbations
            total_pixels = image_array.size
            num_random_pixels = int((percent / 100) * total_pixels)
            
            # More efficient random selection
            flat_indices = np.random.choice(total_pixels, num_random_pixels, replace=False)
            random_indices = np.unravel_index(flat_indices, image_array.shape)
            
            random_perturbed = image_array.copy()
            random_perturbed[random_indices] = 0
            perturbations[f'random{percent}_blackout'] = random_perturbed
    
    return perturbations


def process_single_image(args_tuple):
    """
    Process a single image with all perturbations.
    Designed for multiprocessing.
    """
    image_path, percents, include_random, source_dir, output_dir, random_seed = args_tuple
    
    try:
        # Parse modulation type and SNR
        modulation_type, snr = parse_modulation_and_snr(image_path)
        
        # Setup output directory
        relative_path = os.path.relpath(os.path.dirname(image_path), source_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        
        # Create all perturbations efficiently
        perturbations = create_perturbations(image_array, percents, include_random, random_seed)
        
        # Save all perturbations
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        for pert_name, pert_array in perturbations.items():
            filename = f"{base_filename}_{pert_name}.png"
            output_path = os.path.join(output_subdir, filename)
            Image.fromarray(pert_array).save(output_path)
        
        return f"✓ {modulation_type} @ SNR {snr}: {len(perturbations)} perturbations"
        
    except Exception as e:
        return f"✗ {image_path}: Error - {str(e)}"


def process_batch_multiprocessing(image_paths, percents, include_random, source_dir, output_dir, 
                                num_workers=None, random_seed=42):
    """
    Process images in parallel using multiprocessing.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)  # Increased cap for multi-core CPUs
    
    # Prepare arguments for each worker
    worker_args = [(path, percents, include_random, source_dir, output_dir, random_seed) 
                   for path in image_paths]
    
    print(f"Processing {len(image_paths)} images with {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, worker_args),
            total=len(image_paths),
            desc="Processing images"
        ))
    
    # Print summary
    success_count = sum(1 for r in results if r.startswith("✓"))
    print(f"\nCompleted: {success_count}/{len(results)} images processed successfully")
    
    # Print any errors
    errors = [r for r in results if r.startswith("✗")]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized perturbation of constellation images with parallel processing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--percents', type=int, nargs='+', default=[1, 5, 10], 
                       help='Percentages of pixels to blackout for explainability analysis')
    parser.add_argument('--random', action='store_true', default=True,
                       help='Include random perturbations as baseline comparison')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (auto-detects optimal count)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible perturbations')
    parser.add_argument('--source', type=str, default='constellation',
                       help='Source directory containing constellation images')
    parser.add_argument('--output', type=str, default='perturbed_constellations',
                       help='Output directory for perturbed images')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Collect all image file paths
    image_paths = []
    for root, _, files in os.walk(args.source):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No PNG images found in '{args.source}'")
        return
    
    print(f"Found {len(image_paths)} images")
    print(f"Perturbation percentages: {args.percents}")
    print(f"Include random perturbations: {args.random}")
    print(f"Random seed: {args.seed}")
    
    # Calculate total perturbations
    perturbations_per_image = len(args.percents) * 2  # top + bottom
    if args.random:
        perturbations_per_image += len(args.percents)  # + random
    total_outputs = len(image_paths) * perturbations_per_image
    print(f"Will generate {total_outputs} perturbed images ({perturbations_per_image} per source image)")
    
    # Process images with multiprocessing
    process_batch_multiprocessing(
        image_paths, args.percents, args.random, 
        args.source, args.output, args.workers, args.seed
    )


if __name__ == '__main__':
    main()