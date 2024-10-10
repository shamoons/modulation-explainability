# src/convert_to_constellation.py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, List, Optional
from data_loader import get_dataloader
from utils import get_device


def save_constellation_diagram(iq_data: torch.Tensor, modulation_type: str, snr: float, sample_idx: int, output_dir: str) -> None:
    """
    Save a single constellation diagram to the disk.
    Args:
        iq_data (ndarray): I/Q data for a single sample.
        modulation_type (str): Modulation type for the sample.
        snr (float): SNR value for the sample.
        sample_idx (int): Index of the sample (for naming files).
        output_dir (str): Directory to save the plots.
    """
    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    plt.figure(figsize=(6, 6))
    plt.scatter(in_phase, quadrature, s=5, color='blue')
    plt.axis('off')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Ensure snr is properly formatted
    snr = snr.item() if isinstance(snr, torch.Tensor) else snr
    snr_str = f"SNR_{int(snr) if float(snr).is_integer() else snr}"

    # Create directory for modulation and SNR if it doesn't exist
    modulation_dir = os.path.join(output_dir, modulation_type, snr_str)
    os.makedirs(modulation_dir, exist_ok=True)

    # Save the figure without padding and axis
    save_path = os.path.join(modulation_dir, f'sample_{sample_idx}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def group_by_modulation_snr(dataloader: DataLoader, mod2int: Dict[str, int]) -> Tuple[Dict[str, Dict[float, List[torch.Tensor]]], Dict[int, str]]:
    """
    Group the dataset by modulation type and SNR.
    Args:
        dataloader (DataLoader): DataLoader providing batches of I/Q data.
        mod2int (dict): Dictionary mapping modulation types to integers.

    Returns:
        modulation_snr_samples (dict): Grouped data by modulation and SNR.
        int2mod (dict): Reverse mapping of integers to modulation types.
    """
    int2mod = {v: k for k, v in mod2int.items()}
    modulation_snr_samples: Dict[str, Dict[float, List[Any]]] = {mod: {} for mod in int2mod.values()}

    print("Grouping I/Q data by modulation type and SNR...")
    for inputs, labels, snrs in tqdm(dataloader, desc="Loading data"):
        inputs = inputs.numpy()
        labels = labels.numpy()
        snrs = snrs.numpy()

        for iq_data, label, snr in zip(inputs, labels, snrs):
            modulation_type = int2mod[label]

            # Convert snr from numpy array to scalar (float or int)
            snr = snr.item()

            # Group by modulation and SNR
            if snr not in modulation_snr_samples[modulation_type]:
                modulation_snr_samples[modulation_type][snr] = []
            modulation_snr_samples[modulation_type][snr].append(iq_data)

    # Return the grouped data and modulation-int mapping
    return modulation_snr_samples, int2mod


def process_by_modulation_snr(grouped_data: Dict[str, Dict[float, List[torch.Tensor]]], snrs_to_process: Optional[List[float]] = None, output_dir: str = 'constellation') -> None:
    """
    Process I/Q data grouped by modulation type and SNR, saving the constellation diagrams.
    Args:
        grouped_data (dict): Data grouped by modulation type and SNR.
        snrs_to_process (list, optional): List of SNR values to process. If None, process all SNRs.
        output_dir (str): Directory to save the constellation diagrams.
    """
    print("\nProcessing modulation types and SNRs...")
    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            # If snrs_to_process is provided, skip SNRs that are not in the list
            if snrs_to_process is not None and snr not in snrs_to_process:
                continue

            # Determine the output directory
            snr_str = f"SNR_{int(snr) if float(snr).is_integer() else snr}"
            modulation_dir = os.path.join(output_dir, modulation_type, snr_str)

            # Check if the folder exists and has the correct number of samples
            if os.path.exists(modulation_dir):
                existing_files = [f for f in os.listdir(modulation_dir) if f.endswith('.png')]
                if len(existing_files) == len(samples):
                    print(f"Skipping {modulation_type} at SNR {snr}: Already processed ({len(samples)} samples)")
                    continue  # Skip this modulation/SNR set if all files are already saved

            desc = f'Processing {modulation_type} at {snr} dB'
            for sample_idx, iq_data in tqdm(enumerate(samples), desc=desc, total=len(samples)):
                save_constellation_diagram(iq_data, modulation_type, snr, sample_idx, output_dir)


if __name__ == "__main__":
    print("Loading dataset...")

    # Get the device (CUDA, MPS, or CPU) from utils.py
    device = get_device()

    # Load the entire dataset into a DataLoader
    dataloader, mod2int = get_dataloader(batch_size=4096)  # Adjust batch_size based on your memory

    # Group data by modulation and SNR
    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Print sample counts for each modulation and SNR
    print("\nModulation types, SNRs, and sample counts:")
    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            print(f"{modulation_type} at SNR {snr}: {len(samples)} samples")

    # Example SNR list you want to process, set to None to process all
    snrs_to_process = [0, 10, 20]  # Change this to the SNRs you want or set to None

    # Process the dataset based on SNR selection
    process_by_modulation_snr(grouped_data, snrs_to_process, output_dir='constellation')
