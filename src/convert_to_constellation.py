# src/convert_to_constellation.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List
import logging

from data_loader import get_dataloader

# Setup logging with a simplified format
logging.basicConfig(level=logging.INFO, format='%(message)s')


def save_image(image: np.ndarray, file_path: str, cmap: str = 'gray', background: str = 'white') -> None:
    """
    Save an image to the specified file path.
    Args:
        image (np.ndarray): The image array to save.
        file_path (str): The path where the image will be saved.
        cmap (str): The colormap to use ('gray', 'viridis', etc.).
        background (str): Background color, 'white' or 'black'.
    """
    plt.imshow(image, cmap=cmap, origin='lower')
    if background == 'white':
        plt.gca().set_facecolor('white')
    else:
        plt.gca().set_facecolor('black')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_image(iq_data: torch.Tensor, image_size: Tuple[int, int], image_dir: str, image_name: str) -> None:
    """
    Generate a constellation image from the I/Q signal and save it using `save_image`.

    Args:
        iq_data (torch.Tensor): I/Q data (shape = (N, 2) where 2 corresponds to real and imaginary parts).
        image_size (Tuple[int, int]): Size of the image (e.g., (128, 128)).
        image_dir (str): Directory where the image will be saved.
        image_name (str): The name of the image file.
    """
    blk_size = [5, 25, 50]
    c_factor = 5.0 / torch.tensor(blk_size)
    cons_scale = torch.tensor([2.5, 2.5])

    # Set the final image size without extra padding
    image_size_x, image_size_y = image_size

    d_i_y, d_q_x = 2 * cons_scale[0] / image_size[0], 2 * cons_scale[1] / image_size[1]
    d_xy = torch.sqrt(d_i_y ** 2 + d_q_x ** 2)

    # Convert I/Q data to image positions
    sample_x = torch.round((cons_scale[1] - iq_data[:, 1]) / d_q_x).to(torch.int32)  # Imaginary part (Q)
    sample_y = torch.round((cons_scale[0] + iq_data[:, 0]) / d_i_y).to(torch.int32)  # Real part (I)

    # Create pixel centroid grid
    ii, jj = torch.meshgrid(torch.arange(image_size_x), torch.arange(image_size_y), indexing='ij')
    pixel_centroid_real = -cons_scale[0] + d_i_y / 2 + jj * d_i_y
    pixel_centroid_imag = cons_scale[1] - d_q_x / 2 - ii * d_q_x

    image_array = torch.zeros((image_size_x, image_size_y, 3))

    for kk, blk in enumerate(blk_size):
        blk_x_min = sample_x - blk
        blk_x_max = sample_x + blk + 1
        blk_y_min = sample_y - blk
        blk_y_max = sample_y + blk + 1

        valid = (blk_x_min >= 0) & (blk_y_min >= 0) & (blk_x_max < image_size_x) & (blk_y_max < image_size_y)

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

    # Clip and rescale to 0-255
    image_array = (image_array * 255).numpy().astype(np.uint8)

    # Save the image using `save_image`
    save_image(image_array, os.path.join(image_dir, f"{image_name}.png"), cmap='gray', background='white')


def process_sample(iq_data: np.ndarray, modulation_type: str, snr: float, sample_idx: int, output_dir: str, image_size: Tuple[int, int]) -> None:
    """
    Process a single I/Q data sample through all steps.
    """
    image_name = f"{modulation_type}_SNR_{int(snr)}_sample_{sample_idx}"
    image_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
    os.makedirs(image_dir, exist_ok=True)

    # Convert to torch tensor (without moving to device)
    iq_data_torch = torch.tensor(iq_data)

    generate_image(iq_data_torch, image_size=image_size, image_dir=image_dir, image_name=image_name)


def group_by_modulation_snr(dataloader, mod2int: Dict[str, int]) -> Tuple[Dict[str, Dict[int, List[np.ndarray]]], Dict[int, str]]:
    """
    Group the dataset by modulation type and SNR, ensuring that SNR values are integers.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        mod2int (Dict[str, int]): Mapping of modulation type to integer labels.

    Returns:
        Tuple: A dictionary grouping samples by modulation and SNR, and the reverse mapping from int to mod.
    """
    int2mod = {v: k for k, v in mod2int.items()}
    modulation_snr_samples: Dict[str, Dict[int, List[np.ndarray]]] = {mod: {} for mod in int2mod.values()}

    for inputs, labels, snrs in tqdm(dataloader, desc="Loading data"):
        inputs = inputs.numpy()
        labels = labels.numpy()
        snrs = snrs.numpy()

        for iq_data, label, snr in zip(inputs, labels, snrs):
            modulation_type = int2mod[label]
            snr = int(snr.item())  # Convert SNR to an integer

            if snr not in modulation_snr_samples[modulation_type]:
                modulation_snr_samples[modulation_type][snr] = []
            modulation_snr_samples[modulation_type][snr].append(iq_data)

    logging.info("\nSummary of samples per modulation type and SNR:")
    for modulation_type, snr_dict in modulation_snr_samples.items():
        for snr, samples in snr_dict.items():
            logging.info(f"{modulation_type} at SNR {snr}: {len(samples)} samples")

    return modulation_snr_samples, int2mod


def process_by_modulation_snr(grouped_data: Dict[str, Dict[float, List[np.ndarray]]], output_dir: str = 'constellation', image_size: Tuple[int, int] = (128, 128)) -> None:
    """
    Process I/Q data grouped by modulation type and SNR, saving the constellation diagrams.

    Args:
        grouped_data (Dict): Grouped data dictionary by modulation and SNR.
        output_dir (str): Directory to save the output images.
    """
    logging.info("Starting processing of modulation types and SNRs...")

    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            desc = f'Processing SNR {snr} for {modulation_type}'
            with tqdm(total=len(samples), desc=desc) as pbar:
                for sample_idx, iq_data in enumerate(samples):
                    process_sample(iq_data, modulation_type, snr, sample_idx, output_dir, image_size)
                    pbar.update(1)

    logging.info("Processing completed.")


if __name__ == "__main__":
    # Define the SNRs and modulation types to process
    snrs_to_process = [-20, 10, 0, 10, 20, 30]  # Specify SNRs of interest or set to None for all
    mods_to_process = None  # Specify modulation types of interest or set to None for all
    limit = 5000  # Maximum number of samples to process per modulation-SNR combination

    # Get data loader and define the batch size and SNRs to process
    dataloader, mod2int = get_dataloader(batch_size=4096, snr_list=snrs_to_process, mods_to_process=mods_to_process, limit=limit)

    # Group data based on specified SNRs and modulations
    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Process the data and generate images for the specified SNRs and modulation types
    process_by_modulation_snr(grouped_data, output_dir='constellation')
