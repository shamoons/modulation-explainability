# src/convert_to_constellation.py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_dataloader
from utils import get_device


def save_constellation_batch(iq_batch, mod_batch, snr_batch, batch_idx, output_dir):
    """
    Save a batch of constellation diagrams to the disk.
    Args:
        iq_batch (Tensor): Batch of I/Q data.
        mod_batch (list): List of modulation types corresponding to each sample in the batch.
        snr_batch (Tensor): List of SNR values corresponding to each sample in the batch.
        batch_idx (int): The index of the batch (for naming the files).
        output_dir (str): Directory to save the plots.
    """
    for i, (iq_data, modulation_type, snr) in enumerate(zip(iq_batch, mod_batch, snr_batch)):
        # Convert I/Q data to CPU for processing if using GPU
        iq_data = iq_data.cpu().numpy()

        # Reshape I/Q data if needed
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        in_phase = iq_data[:, 0]  # I component
        quadrature = iq_data[:, 1]  # Q component

        # Reuse the same figure object for faster plotting
        plt.clf()  # Clear the figure for reuse
        plt.scatter(in_phase, quadrature, s=5, color='blue')
        plt.axis('off')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # Ensure snr is properly formatted
        snr = snr.item()
        snr_str = f"SNR_{int(snr) if float(snr).is_integer() else snr}"

        # Create directory for modulation and SNR if it doesn't exist
        modulation_dir = os.path.join(output_dir, modulation_type, snr_str)
        os.makedirs(modulation_dir, exist_ok=True)

        # Save the figure without padding and axis
        save_path = os.path.join(modulation_dir, f'batch_{batch_idx}_sample_{i}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def process_batches(dataloader, mod2int, device, output_dir='constellation'):
    """
    Process batches of I/Q data from the dataloader and save constellation diagrams in batch.
    Args:
        dataloader (DataLoader): DataLoader providing batches of I/Q data.
        mod2int (dict): Dictionary mapping modulation types to integers.
        device (torch.device): The device to use for processing (CPU, CUDA, or MPS).
        output_dir (str): Directory to save the constellation diagrams.
    """
    int2mod = {v: k for k, v in mod2int.items()}

    print("Processing batches of I/Q data...")
    for batch_idx, (inputs, labels, snrs) in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Move data to the processing device (e.g., GPU)
        inputs = inputs.to(device)
        snrs = snrs.to(device)

        # Convert labels to modulation types
        mod_batch = [int2mod[label.item()] for label in labels]

        # Process and save batch of constellation diagrams
        save_constellation_batch(inputs, mod_batch, snrs, batch_idx, output_dir)


if __name__ == "__main__":
    print("Loading dataset...")

    # Get the device (CUDA, MPS, or CPU) from utils.py
    device = get_device()

    # Load the entire dataset into a DataLoader
    dataloader, mod2int = get_dataloader(batch_size=1024)  # Adjust batch_size based on your memory

    # Process the dataset in batches and save constellation diagrams
    process_batches(dataloader, mod2int, device, output_dir='constellation')
