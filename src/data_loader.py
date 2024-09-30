# src/data_loader.py
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data():
    """
    Loads and preprocesses the RadioML 2016.10a dataset.

    The function loads the dataset from a .pkl file, normalizes the signals,
    and splits the data into training, validation, and test sets.

    Returns:
        X_train (ndarray): Training set of signals.
        X_val (ndarray): Validation set of signals.
        X_test (ndarray): Test set of signals.
        y_train (ndarray): Training set labels.
        y_val (ndarray): Validation set labels.
        y_test (ndarray): Test set labels.
    """
    # Use relative path to the dataset from the current script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(
        base_dir, 'data', 'RML2016.10a', 'RML2016.10a_dict.pkl')

    print(f"Loading data file: {dataset_path}")

    # Load the dataset from the .pkl file
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Initialize arrays for signals (X) and labels (y)
    X = []
    y = []

    # Loop through the dictionary and separate signals and labels
    print("Processing signals and labels...")
    for mod_snr, signals in tqdm(data.items(), desc="Loading data"):
        mod_type, snr = mod_snr  # Modulation type and SNR
        X.append(signals)  # Append signal data
        y.extend([mod_type] * len(signals))  # Repeat label for each signal

    # Convert to numpy arrays for easier manipulation
    X = np.vstack(X)  # Stack signals into a single array
    y = np.array(y)  # Convert labels to numpy array

    # Normalize the input signals
    X = X / np.max(np.abs(X), axis=1, keepdims=True)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Example of how to use the function if run as a script
if __name__ == '__main__':
    print("Loading RadioML 2016.10a dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(
        f"Training set: {X_train.shape}, "
        f"Validation set: {X_val.shape}, "
        f"Test set: {X_test.shape}"
    )
