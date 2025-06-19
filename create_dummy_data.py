#!/usr/bin/env python3
"""
Create minimal dummy constellation data for testing the training pipeline.
"""

import os
import numpy as np
from PIL import Image
import torch

def create_dummy_constellation_data():
    """Create minimal dummy constellation images for testing"""
    
    # Create constellation directory structure
    root_dir = "constellation"
    os.makedirs(root_dir, exist_ok=True)
    
    # Test with just 2 modulations and 3 SNR values for speed
    modulations = ["BPSK", "QPSK"] 
    snr_values = [0, 10, 20]  # 3 SNR values from our requested list
    
    print(f"Creating dummy constellation data in '{root_dir}/'...")
    
    total_images = 0
    
    for mod in modulations:
        mod_dir = os.path.join(root_dir, mod)
        os.makedirs(mod_dir, exist_ok=True)
        
        for snr in snr_values:
            snr_dir = os.path.join(mod_dir, f"SNR_{snr}")
            os.makedirs(snr_dir, exist_ok=True)
            
            # Create 8 dummy images per modulation/SNR combination (very small for testing)
            for i in range(8):
                # Create a simple synthetic constellation-like image
                # Different patterns for different modulations
                if mod == "BPSK":
                    # BPSK: two clusters (left and right)
                    img = create_bpsk_pattern(snr)
                else:  # QPSK
                    # QPSK: four clusters (corners)
                    img = create_qpsk_pattern(snr)
                
                # Save as grayscale image
                img_path = os.path.join(snr_dir, f"grayscale_{mod}_SNR_{snr}_{i:03d}.png")
                img_pil = Image.fromarray(img.astype(np.uint8), mode='L')
                img_pil.save(img_path)
                total_images += 1
    
    print(f"✅ Created {total_images} dummy constellation images")
    print(f"   Modulations: {modulations}")
    print(f"   SNR values: {snr_values}")
    print(f"   Images per mod/SNR: 8")
    print(f"   Structure: {root_dir}/MOD/SNR_X/grayscale_*.png")
    
    return root_dir, modulations, snr_values

def create_bpsk_pattern(snr):
    """Create a simple BPSK-like constellation pattern"""
    img = np.zeros((224, 224))
    
    # Add noise based on SNR
    noise_level = max(0.1, 0.5 - snr * 0.02)
    
    # Two main clusters for BPSK
    # Left cluster (around x=75)
    for _ in range(200):
        x = int(75 + np.random.normal(0, 10))
        y = int(112 + np.random.normal(0, 10))
        if 0 <= x < 224 and 0 <= y < 224:
            img[y, x] = min(255, 150 + np.random.normal(0, 20))
    
    # Right cluster (around x=149) 
    for _ in range(200):
        x = int(149 + np.random.normal(0, 10))
        y = int(112 + np.random.normal(0, 10))
        if 0 <= x < 224 and 0 <= y < 224:
            img[y, x] = min(255, 150 + np.random.normal(0, 20))
    
    # Add noise
    noise = np.random.normal(0, noise_level * 50, (224, 224))
    img = np.clip(img + noise, 0, 255)
    
    return img

def create_qpsk_pattern(snr):
    """Create a simple QPSK-like constellation pattern"""
    img = np.zeros((224, 224))
    
    # Add noise based on SNR
    noise_level = max(0.1, 0.5 - snr * 0.02)
    
    # Four clusters for QPSK (corners of a square)
    centers = [(75, 75), (149, 75), (75, 149), (149, 149)]
    
    for cx, cy in centers:
        for _ in range(100):  # 100 points per cluster
            x = int(cx + np.random.normal(0, 8))
            y = int(cy + np.random.normal(0, 8))
            if 0 <= x < 224 and 0 <= y < 224:
                img[y, x] = min(255, 150 + np.random.normal(0, 20))
    
    # Add noise
    noise = np.random.normal(0, noise_level * 50, (224, 224))
    img = np.clip(img + noise, 0, 255)
    
    return img

if __name__ == "__main__":
    create_dummy_constellation_data()