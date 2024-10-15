# src/utils/config_utils.py

import json
import os


def load_loss_config(config_path=None):
    # Set default path to the correct location
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "../config/loss_weights.json")

    # Load the JSON config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['alpha'], config['beta']
