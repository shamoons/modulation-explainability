# src/utils/image_utils.py

from typing import List, Tuple
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb


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


def plot_raw_iq_data(raw_iq_data: torch.Tensor, file_path: str) -> None:
    """
    Plot raw I/Q data and save the image.

    Args:
        raw_iq_data (torch.Tensor): I/Q data as a torch Tensor.
        file_path (str): File path to save the raw plot image.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(raw_iq_data[:, 0].numpy(), raw_iq_data[:, 1].numpy(), c='blue', s=1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


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
        image_types (List[str]): List of image types to generate ('three_channel', 'grayscale', 'raw', 'point').
        raw_iq_data (torch.Tensor): Optional raw I/Q data to save as 'raw' image.
    """
    # Clip and rescale to 0-255 for regular images
    image_array_np = (image_array * 255).numpy().astype(np.uint8)

    for image_type in image_types:
        if image_type == 'grayscale':
            # Combine the three channels into one by averaging
            grayscale_image = np.mean(image_array_np, axis=2).astype(np.uint8)
            pil_image = Image.fromarray(grayscale_image, mode='L')
            resized_image = pil_image.resize(image_size, Image.Resampling.LANCZOS)

            # Prepend image_type to the image_name
            full_image_name = f"{image_type}_{image_name}"
            resized_image.save(os.path.join(image_dir, f"{full_image_name}.png"), format="PNG")

        elif image_type == 'point':
            # Use the image_array directly as it was generated for 'point'

            if image_array_np.ndim == 3:
                image_array_np = np.mean(image_array_np, axis=2)  # Average over the color channels
            plt.figure(figsize=(6, 6))
            plt.imshow(image_array_np.T, origin='lower', cmap='gray', interpolation='nearest')
            plt.axis('off')

            # Save the point-based constellation diagram
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, f"point_{image_name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        elif image_type == 'raw' and raw_iq_data is not None:
            # Plot raw I/Q data as two separate time-series graphs
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

            # Plot In-phase (I) and Quadrature (Q) components separately
            ax1.plot(raw_iq_data[:, 0].numpy(), 'bo')
            ax1.set_title("In-phase")
            ax1.set_xlabel("sample number")
            ax1.set_ylabel("Amplitude")

            ax2.plot(raw_iq_data[:, 1].numpy(), 'bo')
            ax2.set_title("Quadrature")
            ax2.set_xlabel("sample number")
            ax2.set_ylabel("Amplitude")

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, f"raw_{image_name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()


def plot_confusion_matrix(true_labels, pred_labels, label_type, epoch, label_names=None):
    """
    Plot and save a confusion matrix.

    Args:
        true_labels (list of int): True class labels.
        pred_labels (list of int): Predicted class labels.
        label_type (str): Type of label ('Modulation' or 'SNR').
        epoch (int): Current epoch number.
        label_names (list of str or None): List of label names corresponding to class indices.
    """
    # Create directory for confusion matrices if it doesn't exist
    save_dir = "confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))

    # Check if label names are provided, otherwise use numeric labels
    if label_names is None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xticks([])
        plt.yticks([])
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    plt.xlabel(f"Predicted {label_type} Labels")
    plt.ylabel(f"True {label_type} Labels")
    plt.title(f"{label_type} Confusion Matrix - Epoch {epoch + 1}")
    plt.tight_layout()

    # Save confusion matrix
    file_path = os.path.join(save_dir, f"{label_type}_epoch_{epoch + 1}.png")
    plt.savefig(file_path)
    plt.close()

    # Log to Weights and Biases
    wandb.log({f"Confusion Matrix {label_type} Epoch {epoch + 1}": wandb.Image(file_path)})
