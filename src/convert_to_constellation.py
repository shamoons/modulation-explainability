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


def save_enhanced_constellation_image(iq_data: torch.Tensor, modulation_type: str, snr: float, sample_idx: int, output_dir: str) -> None:
    """
    Save an enhanced constellation diagram (three-channel image) to the disk.
    Args:
        iq_data (ndarray): I/Q data for a single sample.
        modulation_type (str): Modulation type for the sample.
        snr (float): SNR value for the sample.
        sample_idx (int): Index of the sample (for naming files).
        output_dir (str): Directory to save the plots.
    """
    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    # Create a 2D histogram (constellation map) to represent sample density
    heatmap, xedges, yedges = np.histogram2d(in_phase, quadrature, bins=256, range=[[-3, 3], [-3, 3]])

    # Normalize heatmap to create intensity values for gray image
    heatmap = heatmap / heatmap.max()

    # Convert to three-channel image by applying different levels of decay
    channel1 = gaussian_filter(heatmap, sigma=1)
    channel2 = gaussian_filter(heatmap, sigma=2)
    channel3 = gaussian_filter(heatmap, sigma=3)
    three_channel_image = np.stack([channel1, channel2, channel3], axis=-1)

    # Save the three-channel image only
    modulation_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
    os.makedirs(modulation_dir, exist_ok=True)

    # Save three-channel image
    plt.imsave(os.path.join(modulation_dir, f'three_channel_sample_{sample_idx}.png'), three_channel_image)


def group_by_modulation_snr(dataloader: DataLoader, mod2int: Dict[str, int], samples_per_modulation: int = 4096) -> Tuple[Dict[str, Dict[float, List[torch.Tensor]]], Dict[int, str]]:
    """
    Group the dataset by modulation type and SNR.
    Args:
        dataloader (DataLoader): DataLoader providing batches of I/Q data.
        mod2int (dict): Dictionary mapping modulation types to integers.
        samples_per_modulation (int): The number of samples to extract per modulation class.

    Returns:
        modulation_snr_samples (dict): Grouped data by modulation and SNR.
        int2mod (dict): Reverse mapping of integers to modulation types.
    """
    int2mod = {v: k for k, v in mod2int.items()}
    modulation_snr_samples: Dict[str, Dict[float, List[Any]]] = {mod: {} for mod in int2mod.values()}

    sample_count = {mod: 0 for mod in int2mod.values()}

    for inputs, labels, snrs in tqdm(dataloader, desc="Loading data"):
        inputs = inputs.numpy()
        labels = labels.numpy()
        snrs = snrs.numpy()

        for iq_data, label, snr in zip(inputs, labels, snrs):
            modulation_type = int2mod[label]

            if sample_count[modulation_type] >= samples_per_modulation:
                continue

            snr = snr.item()

            if snr not in modulation_snr_samples[modulation_type]:
                modulation_snr_samples[modulation_type][snr] = []
            modulation_snr_samples[modulation_type][snr].append(iq_data)

            sample_count[modulation_type] += 1

    return modulation_snr_samples, int2mod


def process_by_modulation_snr(grouped_data: Dict[str, Dict[float, List[torch.Tensor]]], snrs_to_process: Optional[List[float]] = None, output_dir: str = 'constellation') -> None:
    """
    Process I/Q data grouped by modulation type and SNR, saving the constellation diagrams.
    Args:
        grouped_data (dict): Data grouped by modulation type and SNR.
        snrs_to_process (list, optional): List of SNR values to process. If None, process all SNRs.
        output_dir (str): Directory to save the constellation diagrams.
    """
    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            if snrs_to_process is not None and snr not in snrs_to_process:
                continue

            desc = f'Processing {modulation_type} at {snr} dB'
            for sample_idx, iq_data in tqdm(enumerate(samples), desc=desc, total=len(samples)):
                save_enhanced_constellation_image(iq_data, modulation_type, snr, sample_idx, output_dir)


if __name__ == "__main__":
    device = get_device()

    dataloader, mod2int = get_dataloader(batch_size=4096)

    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int, samples_per_modulation=4096)

    process_by_modulation_snr(grouped_data, snrs_to_process=[30], output_dir='constellation')
