#!/usr/bin/env python3
"""
Convert I/Q data to constellation images with optimizations:
1. Multiprocessing for parallel processing of modulation/SNR combinations
2. Batch processing with vectorized operations
3. Optimized image generation using NumPy instead of matplotlib
4. Memory-mapped HDF5 reading for large files
5. Optional GPU acceleration for histogram computation (CUDA/MPS)
"""

import logging
import argparse
import numpy as np
import os
import h5py
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
import torch
from functools import partial
import warnings
import sys
sys.path.append('src')
from utils.device_utils import get_device

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert I/Q data to constellation images.')
    
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process.')
    parser.add_argument('--snr_list', type=str, default=None, help='Comma-separated list of SNRs to process.')
    parser.add_argument('--mod_list', type=str, default=None, help='Comma-separated list of modulation types to process.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing (larger = faster).')
    parser.add_argument('--image_types', type=str, default='grayscale', help='Comma-separated list of image types.')
    parser.add_argument('--h5_dir', type=str, default='data/split_hdf5', help='Directory containing split HDF5 files.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count).')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for histogram computation if available.')
    
    return parser.parse_args()


def fast_histogram_batch(iq_batch, bins=224, range_val=2.5):
    """
    Fast histogram computation for a batch of I/Q samples using vectorized operations.
    
    Args:
        iq_batch: Shape (batch_size, num_samples, 2) - I/Q data
        bins: Number of bins for the histogram
        range_val: Range for I/Q values (-range_val to +range_val)
    
    Returns:
        heatmaps: Shape (batch_size, bins, bins) - constellation diagrams
    """
    batch_size = iq_batch.shape[0]
    heatmaps = np.zeros((batch_size, bins, bins), dtype=np.float32)
    
    # Process each sample in the batch
    for i in range(batch_size):
        # Extract I and Q components
        I = iq_batch[i, :, 0]
        Q = iq_batch[i, :, 1]
        
        # Fast 2D histogram using NumPy
        H, _, _ = np.histogram2d(I, Q, bins=bins, range=[[-range_val, range_val], [-range_val, range_val]])
        
        # Transpose to match expected orientation
        H = H.T
        
        # Normalize using fast operations
        if H.max() > 0:
            H = H / H.max()
        
        heatmaps[i] = H
    
    return heatmaps



# Global device cache to avoid repeated detection
_cached_device = None

def get_cached_device():
    """Get device with caching to avoid repeated prints."""
    global _cached_device
    if _cached_device is None:
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            _cached_device = get_device()
    return _cached_device


def fast_histogram_batch_gpu(iq_batch, bins=224, range_val=2.5):
    """
    GPU-accelerated histogram computation using PyTorch (supports CUDA and MPS).
    
    Args:
        iq_batch: Shape (batch_size, num_samples, 2) - I/Q data
        bins: Number of bins for the histogram
        range_val: Range for I/Q values
    
    Returns:
        heatmaps: Shape (batch_size, bins, bins) - constellation diagrams
    """
    device = get_cached_device()
    
    # Convert to torch tensor
    iq_tensor = torch.from_numpy(iq_batch).to(device)
    batch_size = iq_tensor.shape[0]
    
    # Create output tensor
    heatmaps = torch.zeros((batch_size, bins, bins), device=device, dtype=torch.float32)
    
    # Compute bin edges
    edges = torch.linspace(-range_val, range_val, bins + 1, device=device)
    
    for i in range(batch_size):
        I = iq_tensor[i, :, 0]
        Q = iq_tensor[i, :, 1]
        
        # Digitize the data (find which bin each point belongs to)
        I_bins = torch.bucketize(I, edges) - 1
        Q_bins = torch.bucketize(Q, edges) - 1
        
        # Clamp to valid range
        I_bins = torch.clamp(I_bins, 0, bins - 1)
        Q_bins = torch.clamp(Q_bins, 0, bins - 1)
        
        # Create 2D histogram using scatter_add
        indices = Q_bins * bins + I_bins
        ones = torch.ones_like(indices, dtype=torch.float32)
        flat_hist = torch.zeros(bins * bins, device=device, dtype=torch.float32)
        flat_hist.scatter_add_(0, indices, ones)
        
        # Reshape and normalize
        H = flat_hist.view(bins, bins)
        if H.max() > 0:
            H = H / H.max()
        
        heatmaps[i] = H
    
    return heatmaps.cpu().numpy()


def save_images_batch(heatmaps, output_dir, modulation_type, snr_value, start_idx, image_type='grayscale'):
    """
    Save a batch of constellation diagrams as images.
    
    Args:
        heatmaps: Shape (batch_size, 224, 224) - normalized constellation diagrams
        output_dir: Base output directory
        modulation_type: Modulation type string
        snr_value: SNR value
        start_idx: Starting index for naming
        image_type: Type of image to save
    """
    batch_size = heatmaps.shape[0]
    
    # Create output directory
    image_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr_value)}")
    os.makedirs(image_dir, exist_ok=True)
    
    # Convert to uint8
    heatmaps_uint8 = (heatmaps * 255).astype(np.uint8)
    
    # Save each image
    for i in range(batch_size):
        image_name = f"{image_type}_{modulation_type}_SNR_{int(snr_value)}_sample_{start_idx + i}.png"
        image_path = os.path.join(image_dir, image_name)
        
        # Use PIL for fast saving
        img = Image.fromarray(heatmaps_uint8[i], mode='L')
        img.save(image_path, 'PNG', optimize=False)  # optimize=False for speed


def process_modulation_snr_set(args):
    """
    Process a single modulation/SNR combination.
    
    Args:
        args: Tuple of (modulation_type, snr_value, h5_file_path, batch_size, image_types, use_gpu, limit, device)
    """
    modulation_type, snr_value, h5_file_path, batch_size, image_types, use_gpu, limit, device = args
    
    try:
        # Open HDF5 file with memory mapping for efficiency
        with h5py.File(h5_file_path, 'r') as h5_file:
            X_data = h5_file['X']
            total_samples = X_data.shape[0]
            
            if limit:
                total_samples = min(total_samples, limit)
            
            # Choose histogram function based on device availability
            histogram_func = fast_histogram_batch_gpu if use_gpu and device.type != 'cpu' else fast_histogram_batch
            
            # Process in batches
            pbar = tqdm(range(0, total_samples, batch_size), 
                       desc=f'{modulation_type} @ SNR {snr_value}',
                       leave=False)
            
            for batch_start_idx in pbar:
                batch_end_idx = min(batch_start_idx + batch_size, total_samples)
                actual_batch_size = batch_end_idx - batch_start_idx
                
                # Load batch data (memory-mapped read)
                batch_data = X_data[batch_start_idx:batch_end_idx]
                
                # Generate constellation diagrams
                heatmaps = histogram_func(batch_data)
                
                # Save images
                for image_type in image_types:
                    save_images_batch(heatmaps[:actual_batch_size], 
                                    'constellation', 
                                    modulation_type, 
                                    snr_value, 
                                    batch_start_idx, 
                                    image_type)
        
        return f"✓ {modulation_type} @ SNR {snr_value}: {total_samples} samples"
        
    except Exception as e:
        return f"✗ {modulation_type} @ SNR {snr_value}: Error - {str(e)}"


def main():
    args = parse_args()
    
    # Parse SNR and modulation lists
    if args.snr_list is not None:
        snrs_to_process = [int(snr) for snr in args.snr_list.split(',')]
    else:
        snrs_to_process = list(range(-20, 32, 2))
    
    if args.mod_list is not None:
        mods_to_process = args.mod_list.split(',')
    else:
        # Auto-detect available modulations from h5_dir, excluding analog by default
        analog_mods = ['AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC', 'FM', 'GMSK', 'OOK']
        mods_to_process = []
        if os.path.exists(args.h5_dir):
            for item in sorted(os.listdir(args.h5_dir)):
                item_path = os.path.join(args.h5_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.') and item not in analog_mods:
                    mods_to_process.append(item)
        
        if not mods_to_process:
            print(f"No digital modulation directories found in {args.h5_dir}")
            return
        
        print(f"Using digital modulations only: {sorted(mods_to_process)}")
        print(f"Excluded analog modulations: {sorted(analog_mods)}")
    
    image_types = args.image_types.split(',')
    
    # Detect available device (suppress device_utils print)
    import io
    from contextlib import redirect_stdout
    
    if args.use_gpu:
        with redirect_stdout(io.StringIO()):
            device = get_device()
        device_name = "CUDA" if device.type == 'cuda' else "MPS" if device.type == 'mps' else "CPU"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    print(f"\nConstellation Generation")
    print(f"="*50)
    print(f"H5 Directory: {args.h5_dir}")
    print(f"Modulations: {len(mods_to_process)} types {'(auto-detected)' if args.mod_list is None else ''}")
    print(f"  {', '.join(mods_to_process[:5])}{'...' if len(mods_to_process) > 5 else ''}")
    print(f"SNRs: {len(snrs_to_process)} values from {min(snrs_to_process)} to {max(snrs_to_process)} dB")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device_name} {'(GPU acceleration enabled)' if args.use_gpu and device.type != 'cpu' else ''}")
    print(f"="*50)
    
    # Collect all processing tasks
    tasks = []
    for modulation_type in mods_to_process:
        modulation_dir = os.path.join(args.h5_dir, modulation_type)
        
        if not os.path.exists(modulation_dir):
            continue
        
        for snr_value in snrs_to_process:
            h5_file_path = os.path.join(modulation_dir, f"SNR_{snr_value}", 
                                       f"{modulation_type}_SNR_{snr_value}.h5")
            
            if os.path.isfile(h5_file_path):
                tasks.append((modulation_type, snr_value, h5_file_path, 
                            args.batch_size, image_types, args.use_gpu, args.limit, device))
    
    print(f"\nFound {len(tasks)} modulation/SNR combinations to process")
    
    # Process with multiprocessing
    num_workers = args.num_workers or mp.cpu_count()
    print(f"Using {num_workers} parallel workers\n")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_modulation_snr_set, tasks),
            total=len(tasks),
            desc="Overall progress"
        ))
    
    # Print summary
    print("\n" + "="*50)
    print("Processing Summary:")
    print("="*50)
    for result in results:
        print(result)
    
    print(f"\nCompleted! Images saved to: constellation/")


if __name__ == "__main__":
    main()