# src/utils/constellation_data_processing_utils.py
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import logging

from tqdm import tqdm
from utils.image_utils import generate_and_save_images


def plot_points(iq_data: torch.Tensor, image_size: tuple) -> torch.Tensor:
    """
    Generate the constellation diagram from I/Q data.
    The intensity of points increases when multiple points overlap, and the resulting tensor is returned.

    Args:
        iq_data (torch.Tensor): I/Q data (shape = (N, 2)).
        image_size (tuple): Size of the plot (e.g., (224, 224)).

    Returns:
        torch.Tensor: A grayscale intensity array with the same format as get_image_array output.
    """
    # Create an empty tensor for storing intensities
    image_array = torch.zeros((image_size[0], image_size[1]))

    # Calculate histogram (binning points into the specified image size)
    heatmap, xedges, yedges = np.histogram2d(iq_data[:, 0].numpy(), iq_data[:, 1].numpy(), bins=image_size)

    # Normalize intensities
    heatmap = torch.tensor(heatmap / heatmap.max())

    # Replicate the grayscale values across the 3 channels (to maintain compatibility with existing code structure)
    image_array = heatmap

    # # Plot the points as a grayscale image
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(heatmap.T, origin='lower', cmap='gray', interpolation='nearest')
    # ax.set_xlabel('In-Phase (I)')
    # ax.set_ylabel('Quadrature (Q)')
    # ax.set_title('Constellation Diagram (Grayscale)')

    # plt.show()

    return image_array


def get_image_array(iq_data: torch.Tensor, image_size: tuple) -> torch.Tensor:
    """
    Generate the image array from the I/Q data for a batch of samples.

    Args:
        iq_data (torch.Tensor): I/Q data (shape = (batch_size, N, 2)).
        image_size (tuple): Final size of the image (e.g., (224, 224)).

    Returns:
        torch.Tensor: The image array (batch of images).
    """
    batch_size = iq_data.shape[0]

    # blk_size and scaling factors
    blk_size = [2, 10, 25]
    c_factor = 5.0 / torch.tensor(blk_size)
    cons_scale = torch.tensor([2.5, 2.5])

    internal_image_size_x = image_size[0] + 2 * max(blk_size)
    internal_image_size_y = image_size[1] + 2 * max(blk_size)

    d_i_y = 2 * cons_scale[0] / internal_image_size_x
    d_q_x = 2 * cons_scale[1] / internal_image_size_y
    d_xy = torch.sqrt(d_i_y ** 2 + d_q_x ** 2)

    # Convert I/Q data to image positions for the entire batch
    sample_x = torch.round((cons_scale[1] - iq_data[:, :, 1]) / d_q_x).to(torch.int32)  # Imaginary part (Q)
    sample_y = torch.round((cons_scale[0] + iq_data[:, :, 0]) / d_i_y).to(torch.int32)  # Real part (I)

    # Create a batch of pixel centroid grids
    ii, jj = torch.meshgrid(
        torch.arange(internal_image_size_x),
        torch.arange(internal_image_size_y),
        indexing='ij'
    )

    pixel_centroid_real = -cons_scale[0] + d_i_y / 2 + jj * d_i_y
    pixel_centroid_imag = cons_scale[1] - d_q_x / 2 - ii * d_q_x

    # Pre-allocate a batch of image arrays
    image_array = torch.zeros((batch_size, internal_image_size_x, internal_image_size_y, 3))

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

        for b in range(batch_size):
            for i in torch.where(valid[b])[0]:
                x_min, x_max = blk_x_min[b, i], blk_x_max[b, i]
                y_min, y_max = blk_y_min[b, i], blk_y_max[b, i]

                real_part = iq_data[b, i, 0]
                imag_part = iq_data[b, i, 1]

                real_part_distance = torch.abs(real_part - pixel_centroid_real[x_min:x_max, y_min:y_max])
                imag_part_distance = torch.abs(imag_part - pixel_centroid_imag[x_min:x_max, y_min:y_max])

                sample_distance = torch.sqrt(real_part_distance**2 + imag_part_distance**2)
                image_array[b, x_min:x_max, y_min:y_max, kk] += torch.exp(-c_factor[kk] * sample_distance / d_xy)

            image_array[b, :, :, kk] /= torch.max(image_array[b, :, :, kk])

    return image_array


def renormalize_image(image_array: torch.Tensor) -> torch.Tensor:
    """
    Renormalize the image array to ensure the brightest points are max bright.

    Args:
        image_array (torch.Tensor): The image array where pixel intensities are calculated.

    Returns:
        torch.Tensor: Renormalized image array.
    """
    # Find the maximum value in the image array
    max_value = torch.max(image_array)

    # Avoid division by zero and scale the array to have max intensity as 1.0
    if max_value > 0:
        image_array = image_array / max_value

    return image_array


def process_samples(
    iq_data: np.ndarray, modulation_type: str, snr: float,
    start_idx: int, output_dir: str, image_size: tuple, image_types: list
) -> None:
    """
    Process a batch of I/Q data samples through all steps.

    Args:
        iq_data (np.ndarray): The I/Q data samples (batch of samples).
        modulation_type (str): Modulation type of the samples.
        snr (float): Signal-to-noise ratio of the samples.
        start_idx (int): Starting index for the batch of samples.
        output_dir (str): Directory where images will be saved.
        image_size (tuple): Size of the output images.
        image_types (list): List of image types to generate.
    """
    batch_size = iq_data.shape[0]

    iq_data_torch = torch.tensor(iq_data)

    # Choose the image generation method based on the image type
    if 'point' in image_types:
        image_array = plot_points(iq_data_torch, image_size)
    else:
        image_array = get_image_array(iq_data_torch, image_size)

    image_array = renormalize_image(image_array)

    for i in range(batch_size):
        image_name = f"{modulation_type}_SNR_{int(snr)}_sample_{start_idx + i}"
        image_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
        os.makedirs(image_dir, exist_ok=True)

        # Generate and save images
        generate_and_save_images(image_array[i], image_size, image_dir, image_name, image_types, raw_iq_data=iq_data_torch[i])


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
