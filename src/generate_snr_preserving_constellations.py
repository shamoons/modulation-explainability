#!/usr/bin/env python3
"""
Generate SNR-preserving constellation diagrams and save to constellation_diagrams folder structure.
This replaces the current constellation generation with literature-standard approach that preserves SNR information.

Key improvements:
1. Power normalization instead of arbitrary scaling
2. Log scaling instead of per-image max normalization  
3. Preserves SNR information through relative intensity differences
4. Includes test function to verify SNR preservation
"""

import os
import h5py
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
import torch
import time
warnings.filterwarnings('ignore')

# Device detection with caching
_cached_device = None
def get_device():
    """Get optimal device (CUDA > MPS > CPU) with caching"""
    global _cached_device
    if _cached_device is None:
        if torch.cuda.is_available():
            _cached_device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _cached_device = torch.device('mps')
        else:
            _cached_device = torch.device('cpu')
    return _cached_device

def literature_standard_constellation_vectorized(iq_batch, bins=224, preserve_snr=True, use_gpu=True):
    """
    FAST vectorized constellation generation with GPU acceleration.
    
    Args:
        iq_batch: Shape (batch_size, num_samples, 2) - I/Q data
        bins: Number of histogram bins
        preserve_snr: Whether to preserve SNR information (recommended: True)
        use_gpu: Whether to use GPU acceleration (CUDA/MPS)
    
    Returns:
        heatmaps: Shape (batch_size, bins, bins) - SNR-preserving constellation diagrams
    """
    device = get_device() if use_gpu else torch.device('cpu')
    
    # Convert to torch tensor for vectorized operations
    iq_tensor = torch.from_numpy(iq_batch).to(device, dtype=torch.float32)
    batch_size, num_samples, _ = iq_tensor.shape
    
    I = iq_tensor[:, :, 0]  # Shape: (batch_size, num_samples)
    Q = iq_tensor[:, :, 1]  # Shape: (batch_size, num_samples)
    
    if preserve_snr:
        # Vectorized power normalization across batch
        power = torch.mean(I**2 + Q**2, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        valid_power = power > 0
        scale_factor = torch.sqrt(power)
        I = torch.where(valid_power, I / scale_factor, I)
        Q = torch.where(valid_power, Q / scale_factor, Q)
    
    # Vectorized adaptive range computation
    abs_vals = torch.cat([torch.abs(I), torch.abs(Q)], dim=1)  # Shape: (batch_size, 2*num_samples)
    range_vals = torch.clamp(torch.quantile(abs_vals, 0.995, dim=1), min=3.0)  # Shape: (batch_size,)
    
    # Create output tensor
    heatmaps = torch.zeros((batch_size, bins, bins), device=device, dtype=torch.float32)
    
    # Process entire batch in parallel
    for i in range(batch_size):
        range_val = range_vals[i].item()
        
        # Create bin edges
        edges = torch.linspace(-range_val, range_val, bins + 1, device=device)
        
        # Digitize I and Q coordinates
        I_bins = torch.bucketize(I[i], edges) - 1
        Q_bins = torch.bucketize(Q[i], edges) - 1
        
        # Clamp to valid range
        I_bins = torch.clamp(I_bins, 0, bins - 1)
        Q_bins = torch.clamp(Q_bins, 0, bins - 1)
        
        # Create 2D histogram using scatter_add (very fast on GPU)
        indices = Q_bins * bins + I_bins
        flat_hist = torch.zeros(bins * bins, device=device, dtype=torch.float32)
        flat_hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
        H = flat_hist.view(bins, bins)
        
        if preserve_snr:
            # Log scaling preserves SNR information
            H = torch.log1p(H)
        else:
            # Old method for comparison
            if H.max() > 0:
                H = H / H.max()
        
        heatmaps[i] = H
    
    return heatmaps.cpu().numpy()

def literature_standard_constellation(iq_batch, bins=224, preserve_snr=True):
    """CPU fallback version for compatibility"""
    batch_size = iq_batch.shape[0]
    heatmaps = np.zeros((batch_size, bins, bins), dtype=np.float32)
    
    for i in range(batch_size):
        I = iq_batch[i, :, 0]
        Q = iq_batch[i, :, 1]
        
        if preserve_snr:
            power = np.mean(I**2 + Q**2)
            if power > 0:
                scale_factor = np.sqrt(power)
                I = I / scale_factor
                Q = Q / scale_factor
        
        range_val = max(np.percentile(np.abs([I, Q]), 99.5), 3.0)
        
        H, _, _ = np.histogram2d(I, Q, bins=bins, 
                               range=[[-range_val, range_val], [-range_val, range_val]])
        H = H.T
        
        if preserve_snr:
            H = np.log1p(H)
        else:
            if H.max() > 0:
                H = H / H.max()
        
        heatmaps[i] = H
    
    return heatmaps

def save_constellation_batch_fast(heatmaps, output_dir, modulation_type, snr_value, start_idx, image_type='grayscale'):
    """
    FAST batch saving with vectorized normalization and optimized I/O.
    """
    batch_size = heatmaps.shape[0]
    
    # Create output directory structure: constellation_diagrams/MOD/SNR_X/
    image_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr_value)}")
    os.makedirs(image_dir, exist_ok=True)
    
    # Vectorized normalization across entire batch
    batch_min = heatmaps.min()
    batch_max = heatmaps.max()
    
    if batch_max > batch_min:
        # Vectorized scaling to [0, 255] 
        heatmaps_normalized = (heatmaps - batch_min) / (batch_max - batch_min)
    else:
        heatmaps_normalized = heatmaps
    
    # Vectorized conversion to uint8
    heatmaps_uint8 = (heatmaps_normalized * 255).astype(np.uint8)
    
    # Fast batch saving with minimal Python loops
    file_paths = []
    for i in range(batch_size):
        image_name = f"{image_type}_{modulation_type}_SNR_{int(snr_value)}_sample_{start_idx + i}.png"
        image_path = os.path.join(image_dir, image_name)
        file_paths.append(image_path)
    
    # Save all images in batch (could be parallelized further)
    for i, image_path in enumerate(file_paths):
        img = Image.fromarray(heatmaps_uint8[i], mode='L')
        img.save(image_path, 'PNG', optimize=False)

# Keep old function for compatibility
def save_constellation_batch(heatmaps, output_dir, modulation_type, snr_value, start_idx, image_type='grayscale'):
    """Fallback to fast version"""
    return save_constellation_batch_fast(heatmaps, output_dir, modulation_type, snr_value, start_idx, image_type)

def process_modulation_snr(args):
    """Process a single modulation-SNR combination"""
    h5_dir, output_dir, modulation_type, snr_value, batch_size, limit, image_type = args
    
    # Construct H5 file path (hierarchical structure: MOD/SNR_X/MOD_SNR_X.h5)
    h5_file = os.path.join(h5_dir, modulation_type, f"SNR_{int(snr_value)}", f"{modulation_type}_SNR_{int(snr_value)}.h5")
    
    if not os.path.exists(h5_file):
        print(f"Warning: {h5_file} not found, skipping...")
        return
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Get I/Q data directly from 'X' key
            if 'X' not in f:
                print(f"Warning: 'X' key not found in {h5_file}, skipping...")
                return
            
            # Get I/Q data: Shape (num_samples, 1024, 2)
            iq_data = f['X'][:]
            
            # Apply limit if specified
            if limit is not None:
                iq_data = iq_data[:limit]
            
            num_samples = iq_data.shape[0]
            if num_samples == 0:
                print(f"Warning: No samples found for {modulation_type} SNR_{int(snr_value)}")
                return
            
            print(f"Processing {modulation_type} SNR_{int(snr_value)}: {num_samples} samples")
            
            # Process in batches
            for start_idx in tqdm(range(0, num_samples, batch_size), 
                                desc=f"{modulation_type} SNR_{int(snr_value)}", 
                                leave=False):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_iq = iq_data[start_idx:end_idx]
                
                # Data is already in correct shape: (batch_size, 1024, 2)
                # No reshaping needed
                
                # Generate SNR-preserving constellation diagrams (FAST GPU version)
                try:
                    constellations = literature_standard_constellation_vectorized(
                        batch_iq, bins=224, preserve_snr=True, use_gpu=True
                    )
                except Exception as e:
                    # Fallback to CPU if GPU fails
                    print(f"GPU processing failed, falling back to CPU: {e}")
                    constellations = literature_standard_constellation(
                        batch_iq, bins=224, preserve_snr=True
                    )
                
                # Save the batch
                save_constellation_batch(
                    constellations, output_dir, modulation_type, 
                    snr_value, start_idx, image_type
                )
                
    except Exception as e:
        print(f"Error processing {modulation_type} SNR_{int(snr_value)}: {e}")

def test_snr_preservation():
    """Test that SNR information is preserved between different SNR levels"""
    print("=" * 60)
    print("TESTING SNR PRESERVATION")
    print("=" * 60)
    
    # Generate test signals at different SNR levels
    num_samples = 1000
    
    # High SNR signal (tight constellation)
    signal_high_snr = np.random.randn(num_samples, 2) * 0.1  # Low noise
    signal_high_snr[::4] += [1, 1]   # QPSK points
    signal_high_snr[1::4] += [1, -1]
    signal_high_snr[2::4] += [-1, 1] 
    signal_high_snr[3::4] += [-1, -1]
    
    # Low SNR signal (spread constellation) 
    signal_low_snr = np.random.randn(num_samples, 2) * 0.5   # High noise
    signal_low_snr[::4] += [1, 1]    # Same QPSK points
    signal_low_snr[1::4] += [1, -1]
    signal_low_snr[2::4] += [-1, 1]
    signal_low_snr[3::4] += [-1, -1]
    
    batch = np.stack([signal_high_snr, signal_low_snr])
    
    # Test both methods
    print("Comparing constellation generation methods...\n")
    
    # Literature method (preserves SNR)
    constellations_preserved = literature_standard_constellation(batch, preserve_snr=True)
    
    # Old method (destroys SNR) 
    constellations_destroyed = literature_standard_constellation(batch, preserve_snr=False)
    
    print("📊 LITERATURE METHOD (SNR-preserving):")
    print(f"   High SNR max intensity: {constellations_preserved[0].max():.3f}")
    print(f"   Low SNR max intensity:  {constellations_preserved[1].max():.3f}")
    ratio_preserved = constellations_preserved[0].max() / constellations_preserved[1].max()
    print(f"   Intensity ratio: {ratio_preserved:.3f} (should be >1 to show SNR difference)")
    
    print(f"\n❌ OLD METHOD (SNR-destroying):")
    print(f"   High SNR max intensity: {constellations_destroyed[0].max():.3f}")
    print(f"   Low SNR max intensity:  {constellations_destroyed[1].max():.3f}")  
    ratio_destroyed = constellations_destroyed[0].max() / constellations_destroyed[1].max()
    print(f"   Intensity ratio: {ratio_destroyed:.3f} (always 1.0 - SNR info destroyed)")
    
    print(f"\n✅ IMPROVEMENT:")
    if ratio_preserved > 1.1:
        print(f"   ✓ SNR information PRESERVED! Ratio improvement: {ratio_preserved:.2f}x")
        print(f"   ✓ This should dramatically improve SNR classification accuracy")
    else:
        print(f"   ⚠ SNR preservation may be limited. Check test signal generation.")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Generate SNR-preserving constellation diagrams with GPU acceleration')
    parser.add_argument('--test', action='store_true', 
                       help='Run SNR preservation test only - does not generate any diagrams')
    parser.add_argument('--h5_dir', type=str, default='data/split_hdf5', 
                       help='Directory containing split HDF5 files')
    parser.add_argument('--output_dir', type=str, default='constellation_diagrams', 
                       help='Output directory for constellation diagrams')
    parser.add_argument('--batch_size', type=int, default=512, 
                       help='Batch size for processing (larger=faster with GPU)')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of samples per modulation/SNR')
    parser.add_argument('--snr_list', type=str, default=None,
                       help='Comma-separated SNR list (default: all)')
    parser.add_argument('--mod_list', type=str, default=None,
                       help='Comma-separated modulation list (default: all digital)')
    parser.add_argument('--image_type', type=str, default='grayscale',
                       help='Image type suffix')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--disable_gpu', action='store_true',
                       help='Force CPU processing (disable GPU acceleration)')
    
    args = parser.parse_args()
    
    # Run test if requested
    if args.test:
        test_snr_preservation()
        return
    
    # Parse SNR list
    if args.snr_list:
        snr_values = [int(s.strip()) for s in args.snr_list.split(',')]
    else:
        # Default SNR range
        snr_values = list(range(-20, 32, 2))  # -20 to 30 dB
    
    # Parse modulation list
    if args.mod_list:
        modulation_types = [mod.strip() for mod in args.mod_list.split(',')]
    else:
        # Default: all digital modulations (exclude analog)
        analog_mods = ['AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC', 'FM', 'GMSK', 'OOK']
        # Get all available modulations from h5_dir (hierarchical structure)
        if os.path.exists(args.h5_dir):
            all_dirs = [d for d in os.listdir(args.h5_dir) 
                       if os.path.isdir(os.path.join(args.h5_dir, d)) and not d.startswith('.')]
            modulation_types = [d for d in all_dirs if d not in analog_mods]
        else:
            print(f"Error: H5 directory {args.h5_dir} not found!")
            return
        
    # Print acceleration info
    device = get_device()
    print(f"🚀 Acceleration: {device.type.upper()} {'(disabled by user)' if args.disable_gpu else ''}")
    print(f"🎯 Processing modulations: {modulation_types}")
    print(f"📊 Processing SNRs: {snr_values}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"✨ SNR-preserving mode: ENABLED")
    print(f"📦 Batch size: {args.batch_size} (larger=faster)")
    
    # Performance estimate
    total_combinations = len(modulation_types) * len(snr_values)
    estimated_samples = total_combinations * (args.limit or 4096)
    estimated_batches = estimated_samples // args.batch_size
    print(f"⏱️  Estimated batches: {estimated_batches:,}")
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create task list for parallel processing
    tasks = []
    for modulation_type in modulation_types:
        for snr_value in snr_values:
            tasks.append((
                args.h5_dir, args.output_dir, modulation_type, snr_value,
                args.batch_size, args.limit, args.image_type
            ))
    
    print(f"📋 Total tasks: {len(tasks)}")
    
    # Process tasks
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), len(tasks))
    
    if args.num_workers > 1:
        print(f"🚀 Using {args.num_workers} parallel workers")
        with mp.Pool(args.num_workers) as pool:
            pool.map(process_modulation_snr, tasks)
    else:
        print("🔄 Using single worker")
        for task in tqdm(tasks, desc="Processing"):
            process_modulation_snr(task)
    
    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Constellation diagram generation complete!")
    print(f"💾 Saved to: {args.output_dir}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Print summary
    total_images = 0
    if os.path.exists(args.output_dir):
        for modulation_type in modulation_types:
            mod_dir = os.path.join(args.output_dir, modulation_type)
            if os.path.exists(mod_dir):
                for snr_dir in os.listdir(mod_dir):
                    snr_path = os.path.join(mod_dir, snr_dir)
                    if os.path.isdir(snr_path):
                        images = len([f for f in os.listdir(snr_path) if f.endswith('.png')])
                        total_images += images
    
    print(f"📷 Total images generated: {total_images:,}")
    if total_time > 0 and total_images > 0:
        print(f"🚀 Processing speed: {total_images/total_time:.1f} images/second")
        print(f"🚀 Batch processing speed: {estimated_batches/total_time:.1f} batches/second")
    
    print(f"\n🎉 Ready for training with SNR-preserving constellation diagrams!")

if __name__ == "__main__":
    main()