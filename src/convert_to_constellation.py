# src/convert_to_constellation.py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_dataloaders


def save_constellation_diagram(iq_data, modulation_type, snr, sample_idx, output_dir):
    """
    Save a constellation diagram of the I/Q data for a single sample.

    Args:
        iq_data (ndarray): The I/Q data (1024, 2).
        modulation_type (str): The modulation type label for the sample.
        snr (int or str): The SNR value for the sample.
        sample_idx (int): The index of the sample being saved.
        output_dir (str): Directory where the plot will be saved.
    """
    # Reshape if necessary
    if len(iq_data.shape) == 1:
        iq_data = iq_data.reshape(-1, 2)

    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    plt.figure(figsize=(6, 6))
    plt.scatter(in_phase, quadrature, s=1, color='blue')
    plt.axis('off')  # Remove axis for clean plot

    # Check if SNR is available, else set to 'unknown'
    if snr is None:
        snr = 'unknown'

    # Create the directory if it doesn't exist
    modulation_dir = os.path.join(output_dir, modulation_type, f"SNR_{snr}")
    os.makedirs(modulation_dir, exist_ok=True)

    # Save the plot
    save_path = os.path.join(modulation_dir, f'sample_{sample_idx}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved sample {sample_idx} to {save_path}")


def convert_all_to_constellations(train_loader, mod2int, output_dir='constellation'):
    """
    Convert all samples from the training set into constellation diagrams and save them.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        mod2int (dict): Mapping of modulation types to integers.
        output_dir (str): Directory where the constellations will be saved.
    """
    # Reverse the mod2int dictionary to get int2mod mapping
    int2mod = {v: k for k, v in mod2int.items()}

    # Iterate through the entire DataLoader
    print("Converting I/Q data to constellation diagrams...")
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.numpy()  # Convert tensors to numpy arrays
        labels = labels.numpy()

        for sample_idx, (iq_data, label) in enumerate(zip(inputs, labels)):
            modulation_type = int2mod[label]
            # For now, assuming SNR is unavailable; using a placeholder value (can be modified to retrieve real SNR)
            snr = 'unknown'
            save_constellation_diagram(iq_data[0], modulation_type, snr, batch_idx * len(inputs) + sample_idx, output_dir)


if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader, mod2int = get_dataloaders(batch_size=64)  # Load only limited samples for now

    # Convert the training data to constellation diagrams
    convert_all_to_constellations(train_loader, mod2int, output_dir='constellation')
