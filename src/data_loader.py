# src/data_loader.py
import h5py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm


class RadioMLDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for the RadioML 2018.01A dataset.
    This class handles loading and accessing data in batches.
    """

    def __init__(self, X, y):
        """
        Args:
            X (ndarray): The signal data (I/Q components).
            y (ndarray): The labels (modulation types as integers).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(h5_file='data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5', json_file='data/RML2018.01A/classes-fixed.json', limit=None):
    """
    Loads and preprocesses the RadioML 2018.01A dataset from an HDF5 file.
    If a limit is specified, it loads only that many samples; otherwise, it loads the entire dataset.

    Args:
        h5_file (str): Path to the HDF5 file.
        json_file (str): Path to the JSON file mapping modulation types to integers.
        limit (int): Maximum number of samples to load (None means load all data).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, mod2int
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

        # Use tqdm to show the progress of loading large data arrays
        print("Loading I/Q Data...")
        X = np.array([f['X'][i] for i in tqdm(range(limit))])  # I/Q data (num_samples, 1024, 2)

        # Check if the second dimension is 2, if not, reshape
        if X.shape[-1] != 2:
            X = X.reshape(-1, 1024, 2)  # Reshape to (num_samples, 1024, 2)

        print("Loading Labels...")
        Y = np.array([f['Y'][i] for i in tqdm(range(limit))])  # One-hot encoded modulation labels

    print(f"Shape of I/Q data (X): {X.shape}")
    print(f"Shape of labels (Y): {Y.shape}")

    # Convert one-hot encoded labels to integer labels
    y = np.argmax(Y, axis=1)  # Use axis=1 since Y is one-hot encoded

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, mod2int


def get_dataloaders(batch_size=64, h5_file='data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5', json_file='data/RML2018.01A/classes-fixed.json', limit=None):
    """
    Returns DataLoaders for training, validation, and testing sets.

    Args:
        batch_size (int): Number of samples per batch to load.
        h5_file (str): Path to the HDF5 file.
        json_file (str): Path to the JSON file mapping modulation types to integers.
        limit (int): Maximum number of samples to load (None means load all data).

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, mod2int = load_data(h5_file, json_file, limit)

    # Create PyTorch Datasets
    train_dataset = RadioMLDataset(X_train, y_train)
    val_dataset = RadioMLDataset(X_val, y_val)
    test_dataset = RadioMLDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mod2int
