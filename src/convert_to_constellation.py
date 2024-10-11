# src/convert_to_constellation.py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, List, Optional
from data_loader import get_dataloader
from utils import get_device
import logging

# Setup logging with a simplified format
logging.basicConfig(level=logging.INFO, format='%(message)s')


def save_image(image: np.ndarray, file_path: str, cmap: str = 'gray', background: str = 'white') -> None:
    """
    Save an image to the specified file path.
    Args:
        image (np.ndarray): The image array to save.
        file_path (str): The path where the image will be saved.
        cmap (str): The colormap to use ('gray', 'viridis', etc.).
        background (str): Background color, 'white' or 'black'.
    """
    plt.imshow(image, cmap=cmap, origin='lower')
    if background == 'white':
        plt.gca().set_facecolor('white')
    else:
        plt.gca().set_facecolor('black')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_constellation_diagram(iq_data: torch.Tensor, bins: int = 256, save_step: bool = False, modulation_type: str = '', snr: float = 0, sample_idx: int = 0, output_dir: str = '') -> np.ndarray:
    """
    Generate a 2D histogram (constellation diagram) from I/Q data.
    """
    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    # Define the range to match the 7x7 complex plane as described in the paper
    range_min, range_max = -3, 3

    # Create a 2D histogram (constellation map)
    heatmap, _, _ = np.histogram2d(in_phase, quadrature, bins=bins, range=[[range_min, range_max], [range_min, range_max]])

    # Optionally save the constellation diagram
    if save_step:
        modulation_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
        os.makedirs(modulation_dir, exist_ok=True)
        save_image(heatmap, os.path.join(modulation_dir, f'constellation_sample_{sample_idx}.png'), cmap='gray', background='white')

    return heatmap


def generate_gray_image(heatmap: np.ndarray, save_step: bool = False, modulation_type: str = '', snr: float = 0, sample_idx: int = 0, output_dir: str = '') -> np.ndarray:
    """
    Generate a normalized gray image from the constellation diagram.
    """
    gray_image = heatmap / heatmap.max()  # Normalize heatmap

    # Optionally save the gray image
    if save_step:
        modulation_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
        os.makedirs(modulation_dir, exist_ok=True)
        save_image(gray_image, os.path.join(modulation_dir, f'gray_sample_{sample_idx}.png'), cmap='gray', background='black')

    return gray_image


def generate_enhanced_gray_image(gray_image: np.ndarray, save_step: bool = False, modulation_type: str = '', snr: float = 0, sample_idx: int = 0, output_dir: str = '') -> np.ndarray:
    """
    Apply Gaussian filtering to create an enhanced gray image.
    """
    enhanced_gray_image = gaussian_filter(gray_image, sigma=2)

    # Optionally save the enhanced gray image
    if save_step:
        modulation_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
        os.makedirs(modulation_dir, exist_ok=True)
        save_image(enhanced_gray_image, os.path.join(modulation_dir, f'enhanced_gray_sample_{sample_idx}.png'), cmap='gray', background='black')

    return enhanced_gray_image


def generate_three_channel_image(enhanced_gray_image: np.ndarray, save_step: bool = False, modulation_type: str = '', snr: float = 0, sample_idx: int = 0, output_dir: str = '') -> np.ndarray:
    """
    Generate a vibrant three-channel image by applying Gaussian filters with different sigma values.
    """
    # Apply Gaussian filters with different sigma values
    channel1 = gaussian_filter(enhanced_gray_image, sigma=1)
    channel2 = gaussian_filter(enhanced_gray_image, sigma=2)
    channel3 = gaussian_filter(enhanced_gray_image, sigma=3)

    # Normalize each channel to enhance vibrancy
    channel1 = (channel1 - np.min(channel1)) / (np.max(channel1) - np.min(channel1))
    channel2 = (channel2 - np.min(channel2)) / (np.max(channel2) - np.min(channel2))
    channel3 = (channel3 - np.min(channel3)) / (np.max(channel3) - np.min(channel3))

    # Stack the three channels to create a vibrant image
    three_channel_image = np.stack([channel1 * 255, channel2 * 255, channel3 * 255], axis=-1).astype(np.uint8)

    # Optionally save the three-channel image
    if save_step:
        modulation_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
        os.makedirs(modulation_dir, exist_ok=True)
        save_image(three_channel_image, os.path.join(modulation_dir, f'three_channel_sample_{sample_idx}.png'))

    return three_channel_image


def process_sample(iq_data: torch.Tensor, modulation_type: str, snr: float, sample_idx: int, output_dir: str, save_constellation: bool = True, save_gray: bool = True, save_enhanced_gray: bool = True, save_three_channel: bool = True) -> None:
    """
    Process a single I/Q data sample through all steps.
    """
    # Step 1: Generate constellation diagram
    heatmap = generate_constellation_diagram(iq_data, save_step=save_constellation, modulation_type=modulation_type, snr=snr, sample_idx=sample_idx, output_dir=output_dir)

    # Step 2: Generate gray image
    gray_image = generate_gray_image(heatmap, save_step=save_gray, modulation_type=modulation_type, snr=snr, sample_idx=sample_idx, output_dir=output_dir)

    # Step 3: Generate enhanced gray image
    enhanced_gray_image = generate_enhanced_gray_image(gray_image, save_step=save_enhanced_gray, modulation_type=modulation_type, snr=snr, sample_idx=sample_idx, output_dir=output_dir)

    # Step 4: Generate three-channel image
    generate_three_channel_image(enhanced_gray_image, save_step=save_three_channel, modulation_type=modulation_type, snr=snr, sample_idx=sample_idx, output_dir=output_dir)


def group_by_modulation_snr(dataloader: DataLoader, mod2int: Dict[str, int]) -> Tuple[Dict[str, Dict[float, List[torch.Tensor]]], Dict[int, str]]:
    """
    Group the dataset by modulation type and SNR.
    """
    int2mod = {v: k for k, v in mod2int.items()}
    modulation_snr_samples: Dict[str, Dict[float, List[Any]]] = {mod: {} for mod in int2mod.values()}

    for inputs, labels, snrs in tqdm(dataloader, desc="Loading data"):
        inputs = inputs.numpy()
        labels = labels.numpy()
        snrs = snrs.numpy()

        for iq_data, label, snr in zip(inputs, labels, snrs):
            modulation_type = int2mod[label]
            snr = snr.item()

            if snr not in modulation_snr_samples[modulation_type]:
                modulation_snr_samples[modulation_type][snr] = []
            modulation_snr_samples[modulation_type][snr].append(iq_data)

    logging.info("\nSummary of samples per modulation type and SNR:")
    for modulation_type, snr_dict in modulation_snr_samples.items():
        for snr, samples in snr_dict.items():
            logging.info(f"{modulation_type} at SNR {snr}: {len(samples)} samples")

    return modulation_snr_samples, int2mod


def process_by_modulation_snr(grouped_data: Dict[str, Dict[float, List[torch.Tensor]]], snrs_to_process: Optional[List[float]] = None, output_dir: str = 'constellation', save_constellation: bool = True, save_gray: bool = True, save_enhanced_gray: bool = True, save_three_channel: bool = True) -> None:
    """
    Process I/Q data grouped by modulation type and SNR, saving the constellation diagrams.
    """
    logging.info("Starting processing of modulation types and SNRs...")

    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            if snrs_to_process is not None and snr not in snrs_to_process:
                continue

            # Display the processing message for the current SNR and modulation type
            desc = f'Processing SNR {snr} for {modulation_type}'
            with tqdm(total=len(samples), desc=desc) as pbar:
                for sample_idx, iq_data in enumerate(samples):
                    process_sample(iq_data, modulation_type, snr, sample_idx, output_dir, save_constellation, save_gray, save_enhanced_gray, save_three_channel)
                    pbar.update(1)

    logging.info("Processing completed.")


if __name__ == "__main__":
    device = get_device()
    logging.info(f"Using device: {device}")

    dataloader, mod2int = get_dataloader(batch_size=4096)

    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Flags to save intermediate images
    snrs_to_process = [30, 20, 10]
    process_by_modulation_snr(grouped_data, snrs_to_process=snrs_to_process, output_dir='constellation', save_constellation=False, save_gray=False, save_enhanced_gray=False, save_three_channel=True)
