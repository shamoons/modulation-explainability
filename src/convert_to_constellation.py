# src/convert_to_constellation.py
import logging
import argparse
import numpy as np
import os
import h5py
from tqdm import tqdm
from utils.constellation_data_processing_utils import process_samples

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    """
    Parse command-line arguments for SNR, modulation types, limit, batch_size, image_types, and HDF5 directory.
    """
    parser = argparse.ArgumentParser(description='Convert IQ data to constellation images from smaller HDF5 files.')

    # Add arguments for limit, snr_list, mod_list, batch_size, image_types, and HDF5 directory
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process.')
    parser.add_argument('--snr_list', type=str, default=None, help='Comma-separated list of SNRs to process.')
    parser.add_argument('--mod_list', type=str, default=None, help='Comma-separated list of modulation types to process.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for processing.')
    parser.add_argument('--image_types', type=str, default='grayscale', help='Comma-separated list of image types to generate.')
    parser.add_argument('--h5_dir', type=str, default='data/split_hdf5', help='Directory containing the split HDF5 files.')

    return parser.parse_args()


def process_modulation_snr_set(modulation_type, snr_value, h5_file_path, batch_size, image_types):
    """
    Function to process a single modulation/SNR set from an HDF5 file.
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        X_data = h5_file['X'][:]  # I/Q components
        Y_data = h5_file['Y'][:]  # Modulation labels (not used in this script)
        Z_data = h5_file['Z'][:]  # SNR values (not used in this script)

        total_samples = X_data.shape[0]

        tqdm_desc = f'Processing {modulation_type} at SNR {snr_value}'
        for batch_start_idx in tqdm(range(0, total_samples, batch_size), desc=tqdm_desc):
            batch_samples = X_data[batch_start_idx:batch_start_idx + batch_size]
            process_samples(
                np.array(batch_samples), modulation_type, snr_value,
                batch_start_idx, 'constellation_points', (224, 224), image_types
            )


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Convert comma-separated SNR list to a list of integers
    if args.snr_list is not None:
        snrs_to_process = [int(snr) for snr in args.snr_list.split(',')]
    else:
        snrs_to_process = list(range(-20, 32, 2))  # Default SNR values from -20 dB to +30 dB

    # Convert comma-separated modulation list to a list of strings
    if args.mod_list is not None:
        mods_to_process = args.mod_list.split(',')
    else:
        # Process all modulation types by scanning the h5_dir
        mods_to_process = []
        for item in os.listdir(args.h5_dir):
            item_path = os.path.join(args.h5_dir, item)
            if os.path.isdir(item_path):
                mods_to_process.append(item)
        
        if not mods_to_process:
            # Fallback to default list if no directories found
            mods_to_process = ['16APSK', '32APSK', '64APSK', '128APSK', '32QAM', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', 'AM_DSB_SC', 'FM', 'GMSK', 'OQPSK']

    # Define image types, including "raw"
    image_types = args.image_types.split(',')
    print(f"Generating images for the following types: {image_types}")

    # Ensure the HDF5 directory exists
    if not os.path.exists(args.h5_dir):
        raise FileNotFoundError(f"HDF5 directory {args.h5_dir} does not exist!")

    # Iterate over modulation and SNR directories
    for modulation_type in mods_to_process:
        modulation_dir = os.path.join(args.h5_dir, modulation_type)

        if not os.path.exists(modulation_dir):
            print(f"Skipping modulation {modulation_type} because directory doesn't exist.")
            continue

        # Iterate over the SNR values for the modulation
        for snr_value in snrs_to_process:
            snr_dir = os.path.join(modulation_dir, f"SNR_{snr_value}")
            if not os.path.exists(snr_dir):
                print(f"Skipping SNR {snr_value} for modulation {modulation_type} because directory doesn't exist.")
                continue

            # Find the HDF5 file for the current modulation and SNR combination
            h5_file_path = os.path.join(snr_dir, f"{modulation_type}_SNR_{snr_value}.h5")
            if not os.path.isfile(h5_file_path):
                print(f"Skipping HDF5 file {h5_file_path} because it doesn't exist.")
                continue

            # Process the modulation/SNR set without multi-threading
            process_modulation_snr_set(modulation_type, snr_value, h5_file_path, 32, image_types)
