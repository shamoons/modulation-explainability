# src/convert_to_constellation.py
import torch
import numpy as np
import os
from PIL import Image
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


def get_image_array(iq_data: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Generate the image array from the I/Q data.

    Args:
        iq_data (torch.Tensor): I/Q data (shape = (N, 2)).
        image_size (Tuple[int, int]): Final size of the image (e.g., (224, 224)).

    Returns:
        torch.Tensor: The image array.
    """
    blk_size = [5, 25, 50]
    c_factor = 5.0 / torch.tensor(blk_size)
    cons_scale = torch.tensor([2.5, 2.5])

    # Use the exact image size provided for internal processing
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

            real_part_distance = torch.abs(
                real_part - pixel_centroid_real[x_min:x_max, y_min:y_max]
            )
            imag_part_distance = torch.abs(
                imag_part - pixel_centroid_imag[x_min:x_max, y_min:y_max]
            )

            sample_distance = torch.sqrt(real_part_distance**2 + imag_part_distance**2)
            image_array[x_min:x_max, y_min:y_max, kk] += torch.exp(
                -c_factor[kk] * sample_distance / d_xy
            )

        image_array[:, :, kk] /= torch.max(image_array[:, :, kk])

    return image_array


def generate_and_save_images(
    image_array: torch.Tensor,
    image_size: Tuple[int, int],
    image_dir: str,
    image_name: str,
    image_types: List[str],
    raw_iq_data: torch.Tensor = None
) -> None:
    """
    Generate images of different types from the image array and save them.

    Args:
        image_array (torch.Tensor): The image array.
        image_size (Tuple[int, int]): Final size of the image (e.g., (224, 224)).
        image_dir (str): Directory where the images will be saved.
        image_name (str): The base name of the image files.
        image_types (List[str]): List of image types to generate ('three_channel', 'grayscale', 'binary', etc.).
        raw_iq_data (torch.Tensor): Optional raw I/Q data to save as 'raw' image.
    """
    # Clip and rescale to 0-255 once
    image_array_np = (image_array * 255).numpy().astype(np.uint8)

    for image_type in image_types:
        if image_type == 'grayscale':
            # Combine the three channels into one by averaging
            grayscale_image = np.mean(image_array_np, axis=2).astype(np.uint8)
            pil_image = Image.fromarray(grayscale_image, mode='L')
        elif image_type == 'three_channel':
            pil_image = Image.fromarray(image_array_np)
        elif image_type == 'binary':
            # Generate binary image with black dots on white background
            grayscale_image = np.mean(image_array_np, axis=2)
            threshold_value = 50  # Adjust this threshold if needed
            binary_image = (grayscale_image > threshold_value).astype(np.uint8) * 255
            pil_image = Image.fromarray(binary_image, mode='L')
        elif image_type == 'raw' and raw_iq_data is not None:
            # Plot raw I/Q data points
            plt.figure(figsize=(6, 6))
            plt.scatter(raw_iq_data[:, 0].numpy(), raw_iq_data[:, 1].numpy(), c='blue', s=1)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.savefig(os.path.join(image_dir, f"raw_{image_name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
            continue  # Skip the resize and save steps for raw
        else:
            continue  # Skip unknown image types

        resized_image = pil_image.resize(image_size, Image.Resampling.LANCZOS)

        # Prepend image_type to the image_name
        full_image_name = f"{image_type}_{image_name}"

        # Save the resized image
        resized_image.save(os.path.join(image_dir, f"{full_image_name}.png"), format="PNG")


def process_sample(
    iq_data: np.ndarray,
    modulation_type: str,
    snr: float,
    sample_idx: int,
    output_dir: str,
    image_size: Tuple[int, int],
    image_types: List[str]
) -> None:
    """
    Process a single I/Q data sample through all steps.

    Args:
        iq_data (np.ndarray): The I/Q data sample.
        modulation_type (str): Modulation type of the sample.
        snr (float): Signal-to-noise ratio of the sample.
        sample_idx (int): Index of the sample.
        output_dir (str): Directory where images will be saved.
        image_size (Tuple[int, int]): Size of the output images.
        image_types (List[str]): List of image types to generate.
    """
    image_name = f"{modulation_type}_SNR_{int(snr)}_sample_{sample_idx}"
    image_dir = os.path.join(output_dir, modulation_type, f"SNR_{int(snr)}")
    os.makedirs(image_dir, exist_ok=True)

    # Convert to torch tensor (without moving to device)
    iq_data_torch = torch.tensor(iq_data)

    # Generate the image array once
    image_array = get_image_array(iq_data_torch, image_size)

    # Generate and save images for all specified image types, including raw I/Q data if needed
    generate_and_save_images(image_array, image_size, image_dir, image_name, image_types, raw_iq_data=iq_data_torch)


def group_by_modulation_snr(
    dataloader,
    mod2int: Dict[str, int]
) -> Tuple[Dict[str, Dict[int, List[np.ndarray]]], Dict[int, str]]:
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


def process_by_modulation_snr(
    grouped_data: Dict[str, Dict[float, List[np.ndarray]]],
    output_dir: str = 'constellation',
    image_size: Tuple[int, int] = (224, 224),
    image_types: List[str] = ['three_channel']
) -> None:
    """
    Process I/Q data grouped by modulation type and SNR, saving the constellation diagrams.

    Args:
        grouped_data (Dict): Grouped data dictionary by modulation and SNR.
        output_dir (str): Directory to save the output images.
        image_size (Tuple[int, int]): Size of the output images.
        image_types (List[str]): List of image types to generate.
    """
    logging.info(f"Starting processing of modulation types and SNRs for image types {image_types}...")

    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            desc = f'Processing SNR {snr} for {modulation_type}'
            with tqdm(total=len(samples), desc=desc) as pbar:
                for sample_idx, iq_data in enumerate(samples):
                    process_sample(iq_data, modulation_type, snr, sample_idx, output_dir, image_size, image_types)
                    pbar.update(1)

    logging.info(f"Processing completed for image types {image_types}.")


if __name__ == "__main__":
    # Define the SNRs and modulation types to process
    snrs_to_process = [30]  # Specify SNRs of interest or set to None for all
    mods_to_process = [
        'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16QAM', '64QAM', '256QAM'
    ]
    limit = 100  # Maximum number of samples to process per modulation-SNR combination

    # Define the image types to process, including "raw"
    image_types = ['grayscale', 'raw']  # List all desired image types here

    # Get data loader and define the batch size and SNRs to process
    dataloader, mod2int = get_dataloader(
        batch_size=4096,
        snr_list=snrs_to_process,
        mods_to_process=mods_to_process,
        limit=limit
    )

    # Group data based on specified SNRs and modulations
    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Process the data and generate images for the specified SNRs and modulation types
    process_by_modulation_snr(
        grouped_data,
        output_dir='constellation',
        image_size=(224, 224),
        image_types=image_types
    )
