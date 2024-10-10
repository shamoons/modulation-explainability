# src/convert_to_constellation.py
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data_loader import get_dataloaders


def save_constellation_diagram(iq_data, modulation_type, snr, sample_idx, output_dir):
    """
    Save a constellation diagram of the I/Q data for a single sample.

    Args:
        iq_data (ndarray): The I/Q data (1024, 2).
        modulation_type (str): The modulation type label for the sample.
        snr (float): The SNR value for the sample.
        sample_idx (int): The index of the sample being saved.
        output_dir (str): Directory where the plot will be saved.
    """
    if len(iq_data.shape) == 1:
        iq_data = iq_data.reshape(-1, 2)

    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    plt.figure(figsize=(6, 6))
    plt.scatter(in_phase, quadrature, s=5, color='blue')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    if isinstance(snr, np.ndarray):
        snr = snr.item()

    snr_str = f"SNR_{int(snr) if float(snr).is_integer() else snr}"

    modulation_dir = os.path.join(output_dir, modulation_type, snr_str)
    os.makedirs(modulation_dir, exist_ok=True)

    save_path = os.path.join(modulation_dir, f'sample_{sample_idx}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_modulation_snr_constellations(modulation_type, snr, samples, output_dir):
    """
    Generate and save constellation diagrams for all samples of a specific modulation and SNR.
    """
    desc = f'Processing {modulation_type} at {snr} dB'
    for sample_idx, iq_data in enumerate(tqdm(samples, desc=desc, total=len(samples))):
        save_constellation_diagram(iq_data, modulation_type, snr, sample_idx, output_dir)


def convert_all_to_constellations_by_modulation_snr(train_loader, mod2int, output_dir='constellation'):
    """
    Convert all samples from the training set into constellation diagrams and save them,
    grouped by modulation type and SNR.
    """
    int2mod = {v: k for k, v in mod2int.items()}

    # Group the inputs by modulation type and SNR
    modulation_snr_samples = {mod: {} for mod in int2mod.values()}

    print("Grouping I/Q data by modulation type and SNR...")
    for inputs, labels, snrs in tqdm(train_loader, desc="Loading data"):
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

    # Print modulation types, SNRs, and counts
    print("\nModulation types, SNRs, and sample counts:")
    for modulation_type, snr_dict in modulation_snr_samples.items():
        for snr, samples in snr_dict.items():
            print(f"{modulation_type} at SNR {snr}: {len(samples)} samples")

    # Process each modulation type and SNR sequentially
    print("\nProcessing modulation types and SNRs sequentially...")
    for modulation_type, snr_dict in modulation_snr_samples.items():
        for snr, samples in snr_dict.items():
            if len(samples) > 0:
                generate_modulation_snr_constellations(modulation_type, snr, samples, output_dir)


if __name__ == "__main__":
    print("Loading dataset...")
    train_loader, val_loader, test_loader, mod2int = get_dataloaders(batch_size=256)

    # Convert the training data to constellation diagrams, grouped by modulation and SNR
    convert_all_to_constellations_by_modulation_snr(train_loader, mod2int, output_dir='constellation')
