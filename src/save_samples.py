# src/save_samples.py
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_dataloaders


def save_random_samples(train_loader, mod2int, num_samples=5, output_dir='output'):
    """
    Save random samples of I/Q data from the dataset to the output directory, labeled with the classification.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        mod2int (dict): Mapping of modulation types to integers.
        num_samples (int): Number of random samples to save.
        output_dir (str): Directory where the samples will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reverse the mod2int dictionary to get int2mod mapping
    int2mod = {v: k for k, v in mod2int.items()}

    # Convert train_loader to a list for random sampling
    data = list(train_loader)

    for i in range(num_samples):
        # Randomly select a batch and then a random sample from that batch
        batch_idx = random.randint(0, len(data) - 1)
        inputs, labels = data[batch_idx]
        sample_idx = random.randint(0, inputs.shape[0] - 1)

        # Get the I/Q components and the label for the selected sample
        iq_data = inputs[sample_idx].numpy()  # (1, 1024, 2) format
        label = labels[sample_idx].item()
        modulation_type = int2mod[label]

        # Save the plot of the sample's I/Q data
        save_sample_plot(iq_data[0], modulation_type, i, output_dir)


def save_sample_plot(iq_data, modulation_type, sample_idx, output_dir):
    """
    Save a plot of the I/Q data for a single sample.

    Args:
        iq_data (ndarray): The I/Q data (1024, 2).
        modulation_type (str): The modulation type label for the sample.
        sample_idx (int): The index of the sample being saved.
        output_dir (str): Directory where the plot will be saved.
    """
    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    # Create plot
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(in_phase)
    plt.title(f"Sample {sample_idx} - In-phase Component - {modulation_type}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(quadrature)
    plt.title(f"Sample {sample_idx} - Quadrature Component - {modulation_type}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'sample_{sample_idx}_{modulation_type}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved sample {sample_idx} as {save_path}")


if __name__ == "__main__":
    # Load the data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, mod2int = get_dataloaders(batch_size=64)

    # Save random samples to output directory
    save_random_samples(train_loader, mod2int, num_samples=5, output_dir='output')
