# src/utils/snr_utils.py
import json
import os


def get_snr_bucket(snr):
    """
    Get the bucket (low, medium, high) for a given SNR value.

    Args:
        snr (float): The SNR value to be classified.

    Returns:
        str: The bucket ('low', 'medium', 'high') to which the SNR belongs.
    """
    config_path = os.path.join(os.path.dirname(__file__), '../config/snr_buckets.json')

    with open(config_path, 'r') as file:
        config = json.load(file)

    for bucket, range_values in config['buckets'].items():
        if range_values['min'] <= snr <= range_values['max']:
            return bucket

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
