# src/data_loader.py
import h5py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


class RadioMLDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for the RadioML 2018.01A dataset.
    This class handles loading and accessing data in batches.
    """

    def __init__(self, X, y, snr):
        """
        Args:
            X (ndarray): The signal data (I/Q components).
            y (ndarray): The labels (modulation types as integers).
            snr (ndarray): The SNR values for each frame.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.snr = torch.tensor(snr, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snr[idx]


def load_all_data(h5_file='data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5',
                  json_file='data/RML2018.01A/classes-fixed.json',
                  limit=None,
                  snr_list=None,
                  mods_to_process=None):
    """
    Loads and preprocesses the RadioML 2018.01A dataset from an HDF5 file without splitting.
    If a limit is specified, it applies the limit per modulation type and SNR combination.
    Also filters by modulation types first and then by SNR values.
    """
    logging.info(f"Loading data from {h5_file} and {json_file}...")

    # Load the modulation type mappings from JSON
    with open(json_file, 'r') as f:
        modulation_types = json.load(f)

    mod2int = {mod: i for i, mod in enumerate(modulation_types)}

    print(f"Available modulation schemes in dataset: {modulation_types}")

    # Open the HDF5 file and load the data
    with h5py.File(h5_file, 'r') as f:
        num_samples = len(f['X'])
        logging.info(f"Total samples in dataset: {num_samples}")

        # Initialize lists to hold filtered data
        X_filtered, Y_filtered, Z_filtered = [], [], []

        # Filter by modulation types first
        if mods_to_process is not None:
            mod_indices = [mod2int[mod] for mod in mods_to_process if mod in mod2int]
        else:
            mod_indices = range(len(modulation_types))

        # Track number of samples loaded for each modulation/SNR combination
        samples_loaded = {}

        for mod_index in mod_indices:
            for snr_index in range(26):  # Assuming there are 26 SNR values from -20 to +30
                snr_value = -20 + (snr_index * 2)  # Compute actual SNR value

                # Check if the SNR value is in the list
                if snr_list is not None and snr_value not in snr_list:
                    continue

                samples_loaded[(mod_index, snr_value)] = 0  # Initialize the sample count

                for frame_index in range(4096):  # 4096 frames per modulation-SNR combination
                    idx = mod_index * 26 * 4096 + snr_index * 4096 + frame_index

                    # Skip if the limit for this modulation/SNR is reached
                    key = (mod_index, snr_value)
                    if limit is not None and samples_loaded.get(key, 0) >= limit:
                        break

                    X_sample = f['X'][idx]
                    Y_sample = np.argmax(f['Y'][idx])  # One-hot encoded to integer
                    Z_sample = snr_value  # SNR value from computation

                    # Append to filtered lists if the sample passes filtering
                    X_filtered.append(X_sample)
                    Y_filtered.append(Y_sample)
                    Z_filtered.append(Z_sample)

                    samples_loaded[key] += 1

        # After filtering, print the number of samples loaded for each modulation/SNR
        for (mod_idx, snr_val), count in samples_loaded.items():
            mod_name = list(mod2int.keys())[mod_idx]
            print(f"Modulation: {mod_name}, SNR: {snr_val}, Samples loaded: {count}")

        X = np.array(X_filtered)
        Y = np.array(Y_filtered)
        Z = np.array(Z_filtered)

    logging.info(f"Filtered shapes - I/Q data (X): {X.shape}, labels (Y): {Y.shape}, SNR values (Z): {Z.shape}")

    return X, Y, Z, mod2int


def get_dataloader(batch_size=64,
                   h5_file='data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5',
                   json_file='data/RML2018.01A/classes-fixed.json',
                   limit=None,
                   snr_list=None,
                   mods_to_process=None):
    """
    Returns a DataLoader for the entire dataset, optionally filtered by SNR and modulation type.

    Args:
        batch_size (int): Number of samples per batch to load.
        h5_file (str): Path to the HDF5 file.
        json_file (str): Path to the JSON file mapping modulation types to integers.
        limit (int): Maximum number of samples to load (None means load all data).
        snr_list (list of str or None): List of SNR values to load. If None, load all SNRs.
        mods_to_process (list of str or None): List of modulation types to load. If None, load all modulations.

    Returns:
        DataLoader: DataLoader for the entire dataset.
        mod2int: Modulation to integer mapping.
    """
    X, Y, Z, mod2int = load_all_data(h5_file, json_file, limit, snr_list, mods_to_process)

    # Create PyTorch Dataset
    dataset = RadioMLDataset(X, Y, Z)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12, shuffle=True, pin_memory=True)

    logging.info(f"Dataloader created with batch size {batch_size}")

    return dataloader, mod2int


if __name__ == "__main__":
    snr_list = [30]
    # mods_to_process = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', 'QAM16', 'QAM64', 'QAM256']
    mods_to_process = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16QAM', '64QAM', '256QAM']

    # Get DataLoader
    dataloader, mod2int = get_dataloader(batch_size=128, snr_list=snr_list, mods_to_process=mods_to_process)
