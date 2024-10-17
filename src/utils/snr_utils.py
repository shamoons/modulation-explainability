# src/utils/snr_utils.py
import json
import os


def get_snr_bucket(snr):
    config_path = os.path.join(os.path.dirname(__file__), '../config/snr_buckets.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    for idx, (bucket, range_values) in enumerate(config['buckets'].items()):
        if range_values['min'] <= snr <= range_values['max']:
            return idx  # Return the index (0 for low, 1 for medium, 2 for high)

    raise ValueError(f"SNR value {snr} is out of the configured range.")


def get_number_of_snr_buckets():
    """
    Get the number of SNR buckets defined in the config file.

    Returns:
        int: Number of SNR buckets.
    """
    config_path = os.path.join(os.path.dirname(__file__), '../config/snr_buckets.json')

    with open(config_path, 'r') as file:
        config = json.load(file)

    return len(config['buckets'])


def get_snr_label_names():
    """
    Returns the label names for the SNR buckets.
    Example: ["low", "medium", "high"]

    Returns:
        list: List of label names for the SNR buckets.
    """
    config_path = os.path.join(os.path.dirname(__file__), '../config/snr_buckets.json')

    with open(config_path, 'r') as file:
        config = json.load(file)

    return list(config['buckets'].keys())


def get_snr_bucket_label(snr):
    """
    Maps SNR values to their corresponding bucket labels (like "low", "medium", "high").
    """
    config_path = os.path.join(os.path.dirname(__file__), '../config/snr_buckets.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    for bucket_label, range_values in config['buckets'].items():
        if range_values['min'] <= snr <= range_values['max']:
            return bucket_label  # Return the label like "low", "medium", "high"

    raise ValueError(f"SNR value {snr} is out of the configured range.")
