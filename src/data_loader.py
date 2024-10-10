# src/data_loader.py
import h5py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader


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
                  limit=None):
    """
    Loads and preprocesses the RadioML 2018.01A dataset from an HDF5 file without splitting.
    If a limit is specified, it loads only that many samples; otherwise, it loads the entire dataset.

    Args:
        h5_file (str): Path to the HDF5 file.
        json_file (str): Path to the JSON file mapping modulation types to integers.
        limit (int): Maximum number of samples to load (None means load all data).

    Returns:
        X, Y, Z, mod2int: All I/Q data, labels, SNR values, and modulation-to-integer mapping.
    """
    # Load the modulation type mappings from JSON
    with open(json_file, 'r') as f:
        modulation_types = json.load(f)

    mod2int = {mod: i for i, mod in enumerate(modulation_types)}

    # Open the HDF5 file and load the data
    with h5py.File(h5_file, 'r') as f:
        num_samples = len(f['X'])
        if limit is None:
            limit = num_samples  # If no limit is specified, load all data
        else:
            limit = min(limit, num_samples)

        print(f"Total samples in dataset: {num_samples}")
        print(f"Loading {limit} samples...")

        # Directly load I/Q Data, Labels, and SNR
        X = f['X'][:limit]  # I/Q data (num_samples, 1024, 2)
        Y = np.argmax(f['Y'][:limit], axis=1)  # One-hot encoded labels converted to integers
        Z = f['Z'][:limit]  # SNR values

    print(f"Shape of I/Q data (X): {X.shape}")
    print(f"Shape of labels (Y): {Y.shape}")
    print(f"Shape of SNR values (Z): {Z.shape}")

    return X, Y, Z, mod2int


def get_dataloader(batch_size=64,
                   h5_file='data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5',
                   json_file='data/RML2018.01A/classes-fixed.json',
                   limit=None):
    """
    Returns a DataLoader for the entire dataset.

    Args:
        batch_size (int): Number of samples per batch to load.
        h5_file (str): Path to the HDF5 file.
        json_file (str): Path to the JSON file mapping modulation types to integers.
        limit (int): Maximum number of samples to load (None means load all data).

    Returns:
        DataLoader: DataLoader for the entire dataset.
        mod2int: Modulation to integer mapping.
    """
    X, Y, Z, mod2int = load_all_data(h5_file, json_file, limit)

    # Create PyTorch Dataset
    dataset = RadioMLDataset(X, Y, Z)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, mod2int
