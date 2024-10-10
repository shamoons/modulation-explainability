# src/convert_to_constellation.py
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
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


def generate_modulation_constellations(modulation_type, samples, snrs, output_dir):
    """
    Generate and save constellation diagrams for all samples of a specific modulation type.
    """
    # Use tqdm progress bar for the samples being processed
    for sample_idx, (iq_data, snr) in enumerate(tqdm(zip(samples, snrs), total=len(samples), desc=f'Processing {modulation_type}')):
        save_constellation_diagram(iq_data, modulation_type, snr, sample_idx, output_dir)


def convert_all_to_constellations_parallel(train_loader, mod2int, output_dir='constellation'):
    """
    Convert all samples from the training set into constellation diagrams and save them,
    parallelized by modulation type.
    """
    int2mod = {v: k for k, v in mod2int.items()}

    # Group the inputs by modulation type
    modulation_samples = {mod: [] for mod in int2mod.values()}
    modulation_snrs = {mod: [] for mod in int2mod.values()}

    print("Grouping I/Q data by modulation type...")
    for inputs, labels, snrs in tqdm(train_loader, desc="Loading data"):
        inputs = inputs.numpy()
        labels = labels.numpy()
        snrs = snrs.numpy()

        for iq_data, label, snr in zip(inputs, labels, snrs):
            modulation_type = int2mod[label]
            modulation_samples[modulation_type].append(iq_data)
            modulation_snrs[modulation_type].append(snr)

    # Print modulation types and counts
    print("\nModulation types and sample counts:")
    for modulation_type, samples in modulation_samples.items():
        print(f"{modulation_type}: {len(samples)} samples")

    # Parallelize across modulation types
    print("\nStarting parallel processing for modulation types...")
    with ProcessPoolExecutor() as executor:
        futures = []
        for modulation_type in modulation_samples.keys():
            samples = modulation_samples[modulation_type]
            snrs = modulation_snrs[modulation_type]
            if len(samples) > 0:
                futures.append(executor.submit(generate_modulation_constellations, modulation_type, samples, snrs, output_dir))

        # Wait for all processes to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    print("Loading dataset...")
    train_loader, val_loader, test_loader, mod2int = get_dataloaders(batch_size=64)

    # Convert the training data to constellation diagrams in parallel
    convert_all_to_constellations_parallel(train_loader, mod2int, output_dir='constellation')
