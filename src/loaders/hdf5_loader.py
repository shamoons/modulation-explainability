# src/loaders/hdf5_loader.py
import h5py
import os
import json
import numpy as np


class HDF5Loader:
    """
    Class to load and split a large HDF5 file based on modulation and SNR values.
    """

    def __init__(self, input_file, json_file, output_dir, snr_list=None, mod_list=None, frames_per_combination=4096):
        """
        Initialize the loader.

        Args:
            input_file (str): Path to the large HDF5 input file.
            json_file (str): Path to the JSON file containing modulation types.
            output_dir (str): Directory to store the smaller HDF5 files.
            snr_list (list or None): List of SNR values to process. If None, all SNR values will be processed.
            mod_list (list or None): List of modulation types to process. If None, all modulation types will be processed.
            frames_per_combination (int): Number of frames per modulation/SNR combination (default: 4096).
        """
        self.input_file = input_file
        self.json_file = json_file
        self.output_dir = output_dir
        self.snr_list = snr_list
        self.frames_per_combination = frames_per_combination

        # Load modulation types from the JSON file
        self.modulations = self.load_modulation_types()

        # If mod_list is None, process all modulation types
        if mod_list is None:
            self.mod_list = self.modulations
        else:
            self.mod_list = mod_list

        # If snr_list is None, process all SNR values (-20 dB to +30 dB in steps of 2 dB)
        if snr_list is None:
            self.snr_list = list(range(-20, 32, 2))
        else:
            self.snr_list = snr_list

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_modulation_types(self):
        """
        Load modulation types from the JSON file.
        """
        with open(self.json_file, 'r') as f:
            modulations = json.load(f)
        return modulations

    def split_data(self):
        """
        Split the HDF5 data into smaller files based on modulation and SNR values.
        """
        # Open the large HDF5 file for reading
        with h5py.File(self.input_file, 'r') as f:
            X_data = f['X']  # I/Q components
            Y_data = f['Y']  # One-hot encoded modulation types
            Z_data = f['Z']  # SNR values

            total_modulations = len(self.modulations)
            total_snr = len(self.snr_list)

            # Iterate over each modulation type
            for mod_name in self.mod_list:
                mod_idx = self.modulations.index(mod_name)

                # Iterate over each SNR value for the current modulation
                for snr_value in self.snr_list:
                    snr_idx = (snr_value + 20) // 2  # Calculate the SNR index based on -20 to +30 dB

                    start_idx = mod_idx * total_snr * self.frames_per_combination + snr_idx * self.frames_per_combination
                    end_idx = start_idx + self.frames_per_combination

                    # Extract the data for the current modulation/SNR combination
                    X_subset = X_data[start_idx:end_idx]
                    Y_subset = Y_data[start_idx:end_idx]
                    Z_subset = Z_data[start_idx:end_idx]

                    # Create the directory structure for the current modulation and SNR
                    mod_snr_dir = os.path.join(self.output_dir, mod_name, f"SNR_{snr_value}")
                    os.makedirs(mod_snr_dir, exist_ok=True)

                    # Save the extracted data into a smaller HDF5 file
                    output_file = os.path.join(mod_snr_dir, f"{mod_name}_SNR_{snr_value}.h5")
                    with h5py.File(output_file, 'w') as hf_out:
                        hf_out.create_dataset('X', data=X_subset)
                        hf_out.create_dataset('Y', data=Y_subset)
                        hf_out.create_dataset('Z', data=Z_subset)

                    print(f"Saved {mod_name} at SNR {snr_value} to {output_file}")


if __name__ == "__main__":
    # Initialize the HDF5Loader and split the data
    input_file = 'data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5'
    json_file = 'data/RML2018.01A/classes-fixed.json'
    output_dir = 'data/split_hdf5'

    # Run with specific modulation and SNR or process all if None
    snr_list = None  # Process all SNR values if None
    mod_list = None  # Process all modulations if None

    loader = HDF5Loader(input_file, json_file, output_dir, snr_list, mod_list)
    loader.split_data()
