# src/utils/constellation_data_processing_utils.py
from typing import Dict, List, Tuple
import numpy as np
import torch
import os
import logging

from tqdm import tqdm
from utils.image_utils import generate_and_save_images


def get_image_array(iq_data: torch.Tensor, image_size: tuple) -> torch.Tensor:
    """
    Generate the image array from the I/Q data.

    Args:
        iq_data (torch.Tensor): I/Q data (shape = (N, 2)).
        image_size (tuple): Final size of the image (e.g., (224, 224)).

    Returns:
        torch.Tensor: The image array.
    """
    blk_size = [5, 25, 50]
    c_factor = 5.0 / torch.tensor(blk_size)
    cons_scale = torch.tensor([2.5, 2.5])

    internal_image_size_x = image_size[0] + 4 * max(blk_size)
    internal_image_size_y = image_size[1] + 4 * max(blk_size)

    d_i_y = 2 * cons_scale[0] / internal_image_size_x
    d_q_x = 2 * cons_scale[1] / internal_image_size_y
    d_xy = torch.sqrt(d_i_y ** 2 + d_q_x ** 2)

    # Convert I/Q data to image positions
    sample_x = torch.round((cons_scale[1] - iq_data[:, 1]) / d_q_x).to(torch.int32)  # Imaginary part (Q)
    sample_y = torch.round((cons_scale[0] + iq_data[:, 0]) / d_i_y).to(torch.int32)  # Real part (I)

    # Create pixel centroid grid with the internal image size
    ii, jj = torch.meshgrid(
        torch.arange(internal_image_size_x),
        torch.arange(internal_image_size_y),
        indexing='ij'
    )

    pixel_centroid_real = -cons_scale[0] + d_i_y / 2 + jj * d_i_y
    pixel_centroid_imag = cons_scale[1] - d_q_x / 2 - ii * d_q_x

    image_array = torch.zeros((internal_image_size_x, internal_image_size_y, 3))

    for kk, blk in enumerate(blk_size):
        blk_x_min = sample_x - blk
        blk_x_max = sample_x + blk + 1
        blk_y_min = sample_y - blk
        blk_y_max = sample_y + blk + 1

        valid = (
            (blk_x_min >= 0)
            & (blk_y_min >= 0)
            & (blk_x_max < internal_image_size_x)
            & (blk_y_max < internal_image_size_y)
        )

        for i in torch.where(valid)[0]:
            x_min, x_max = blk_x_min[i], blk_x_max[i]
            y_min, y_max = blk_y_min[i], blk_y_max[i]

            real_part = iq_data[i, 0]
            imag_part = iq_data[i, 1]

            real_part_distance = torch.abs(real_part - pixel_centroid_real[x_min:x_max, y_min:y_max])
            imag_part_distance = torch.abs(imag_part - pixel_centroid_imag[x_min:x_max, y_min:y_max])

            sample_distance = torch.sqrt(real_part_distance**2 + imag_part_distance**2)
            image_array[x_min:x_max, y_min:y_max, kk] += torch.exp(-c_factor[kk] * sample_distance / d_xy)

        image_array[:, :, kk] /= torch.max(image_array[:, :, kk])

    return image_array


def process_sample(iq_data: np.ndarray, modulation_type: str, snr: float, sample_idx: int, output_dir: str, image_size: tuple, image_types: list) -> None:
    """
    Process a single I/Q data sample through all steps.

    Args:
        iq_data (np.ndarray): The I/Q data sample.
        modulation_type (str): Modulation type of the sample.
        snr (float): Signal-to-noise ratio of the sample.
        sample_idx (int): Index of the sample.
        output_dir (str): Directory where images will be saved.
        image_size (tuple): Size of the output images.
        image_types (list): List of image types to generate.
    """
    image_name = f"{modulation_type}_SNR_{int(snr)}_sample_{sample_idx}"
    image_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
    os.makedirs(image_dir, exist_ok=True)

    iq_data_torch = torch.tensor(iq_data)

    # Generate the image array once
    image_array = get_image_array(iq_data_torch, image_size)

    # Generate and save images
    generate_and_save_images(image_array, image_size, image_dir, image_name, image_types, raw_iq_data=iq_data_torch)


def group_by_modulation_snr(
    dataloader: torch.utils.data.DataLoader,
    mod2int: Dict[str, int]
) -> Tuple[Dict[str, Dict[int, List[np.ndarray]]], Dict[int, str]]:
    """
    Group the dataset by modulation type and SNR, ensuring that SNR values are integers.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        mod2int (Dict[str, int]): Mapping of modulation type to integer labels.

    Returns:
        Tuple[Dict[str, Dict[int, List[np.ndarray]]], Dict[int, str]]:
            A dictionary grouping samples by modulation and SNR, 
            and the reverse mapping from integer to modulation type.
    """

    # Reverse mapping: Integer to modulation type
    int2mod = {v: k for k, v in mod2int.items()}

    # Dictionary to store samples grouped by modulation type and SNR
    modulation_snr_samples: Dict[str, Dict[int, List[np.ndarray]]] = {mod: {} for mod in int2mod.values()}

    # Iterate over the dataloader batches
    for inputs, labels, snrs in tqdm(dataloader, desc="Loading and grouping data by modulation and SNR"):
        inputs = inputs.numpy()
        labels = labels.numpy()
        snrs = snrs.numpy()

        # Loop over the inputs, labels, and SNRs in the current batch
        for iq_data, label, snr in zip(inputs, labels, snrs):
            modulation_type = int2mod[label]
            snr_int = int(snr)  # Convert SNR to an integer value

            # Ensure a list exists for the current modulation and SNR
            if snr_int not in modulation_snr_samples[modulation_type]:
                modulation_snr_samples[modulation_type][snr_int] = []

            # Append the IQ data to the list for the modulation and SNR
            modulation_snr_samples[modulation_type][snr_int].append(iq_data)

    # Log a summary of the grouped data
    logging.info("\nSummary of samples per modulation type and SNR:")
    for modulation_type, snr_dict in modulation_snr_samples.items():
        for snr, samples in snr_dict.items():
            logging.info(f"{modulation_type} at SNR {snr}: {len(samples)} samples")

    return modulation_snr_samples, int2mod
