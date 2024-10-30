# src/loaders/perturbation_loader.py

import torch
from torchvision import transforms
from loaders.constellation_loader import ConstellationDataset


class PerturbationDataset(ConstellationDataset):
    """
    Dataset class for loading constellation data and applying perturbations based on brightness.
    Black out top brightest or dimmest pixels based on the specified percentage.
    """

    def __init__(self, root_dir, image_type="grayscale", snr_list=None, mods_to_process=None, use_snr_buckets=False, perturb_percentage=5, perturb_type="brightest"):
        """
        Initialize the PerturbationDataset.

        Args:
            root_dir (str): Path to the dataset directory.
            image_type (str): Type of image ('grayscale' or 'RGB').
            snr_list (list, optional): List of SNR values to include.
            mods_to_process (list, optional): List of modulation types to include.
            use_snr_buckets (bool): Whether to use SNR buckets.
            perturb_percentage (int): Percentage of pixels to perturb (blackout).
            perturb_type (str): Type of perturbation ('brightest' or 'dimmest').
        """
        super().__init__(root_dir, image_type, snr_list, mods_to_process, use_snr_buckets)
        self.perturb_percentage = perturb_percentage / 100  # Convert to fraction
        self.perturb_type = perturb_type
        self.transform = transforms.ToTensor()

    def perturb_image(self, image):
        """
        Apply perturbation to the image by blacking out a percentage of the brightest or dimmest pixels.

        Args:
            image (PIL Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Perturbed image.
        """
        image = self.transform(image) if not isinstance(image, torch.Tensor) else image
        flat_image = image.flatten()  # Flatten the image for sorting

        # Determine the number of pixels to perturb based on the percentage
        num_pixels = int(self.perturb_percentage * flat_image.numel())

        if self.perturb_type == "brightest":
            _, top_indices = torch.topk(flat_image, num_pixels)
        elif self.perturb_type == "dimmest":
            _, top_indices = torch.topk(-flat_image, num_pixels)
        else:
            raise ValueError("perturb_type must be 'brightest' or 'dimmest'")

        # Create a mask to blackout the chosen pixels
        mask = torch.ones_like(flat_image)
        mask[top_indices] = 0  # Black out the specified pixels

        # Apply the mask and reshape back to original image dimensions
        perturbed_image = (flat_image * mask).reshape(image.shape)

        return perturbed_image

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset, apply perturbation, and return it along with the label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (perturbed_image, modulation_label, snr_label)
        """
        image, modulation_label, snr_label = super().__getitem__(idx)
        perturbed_image = self.perturb_image(image)

        return perturbed_image, modulation_label, snr_label
