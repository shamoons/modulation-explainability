# src/save_samples.py
import os
import matplotlib.pyplot as plt
from loaders.data_loader import get_dataloaders


def save_all_constellation_diagrams(train_loader, mod2int, output_dir='output'):
    """
    Save constellation diagrams of I/Q data from the dataset to the output directory, labeled with the classification.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        mod2int (dict): Mapping of modulation types to integers.
        output_dir (str): Directory where the samples will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reverse the mod2int dictionary to get int2mod mapping
    int2mod = {v: k for k, v in mod2int.items()}

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        for sample_idx in range(inputs.shape[0]):
            iq_data = inputs[sample_idx].numpy()  # Get the I/Q data (1, 1024, 2) or (1024, 2) format
            label = labels[sample_idx].item()
            modulation_type = int2mod[label]

            # Check if the data is correctly shaped
            if len(iq_data.shape) == 3 and iq_data.shape[-1] == 2:
                iq_data = iq_data[0]  # Remove the first dimension if necessary (from (1, 1024, 2) to (1024, 2))
            elif len(iq_data.shape) == 2 and iq_data.shape[-1] != 2:
                print(f"Shape of iq_data: {iq_data.shape}")
                print("Error: expected iq_data to have 2 columns, but got shape {iq_data.shape}. Skipping.")
                continue  # Skip samples with incorrect shapes

            # Save the constellation diagram
            save_constellation_diagram(iq_data, modulation_type, batch_idx, sample_idx, output_dir)


def save_constellation_diagram(iq_data, modulation_type, batch_idx, sample_idx, output_dir):
    """
    Save a constellation diagram (scatter plot) of the I/Q data for a single sample.

    Args:
        iq_data (ndarray): The I/Q data (1024, 2).
        modulation_type (str): The modulation type label for the sample.
        batch_idx (int): The batch index.
        sample_idx (int): The sample index within the batch.
        output_dir (str): Directory where the plot will be saved.
    """
    in_phase = iq_data[:, 0]  # I component
    quadrature = iq_data[:, 1]  # Q component

    # Create constellation plot
    plt.figure(figsize=(6, 6))
    plt.scatter(in_phase, quadrature, s=5, color='blue')
    plt.title(f"Constellation Diagram - {modulation_type} (Batch {batch_idx}, Sample {sample_idx})")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)

    save_path = os.path.join(output_dir, f'constellation_{batch_idx}_{sample_idx}_{modulation_type}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved constellation diagram for Batch {batch_idx}, Sample {sample_idx} as {save_path}")


if __name__ == "__main__":
    # Load the data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, mod2int = get_dataloaders(batch_size=64)

    # Save all constellation diagrams to the output directory
    save_all_constellation_diagrams(train_loader, mod2int, output_dir='output')
