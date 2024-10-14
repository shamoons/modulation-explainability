# src/utils/device_utils.py
import torch


def get_device():
    """
    Determines the best available device for training.
    Prioritizes CUDA (NVIDIA GPU), then Apple Metal (M1/M2), otherwise CPU.
    """
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Check for Apple Metal (M1/M2)
        print("Apple Metal is available. Using MPS (Metal Performance Shaders).")
        return torch.device("mps")
    else:
        print("Using CPU.")
        return torch.device("cpu")
