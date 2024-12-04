# perturb_constellations.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    source_dir = 'constellation'
    output_dir = 'perturbed_constellations'

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all image file paths
    image_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))

    for image_path in tqdm(image_paths, desc='Processing images'):
        # Compute relative path to preserve directory structure
        relative_path = os.path.relpath(os.path.dirname(image_path), source_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # Load the image
        image = Image.open(image_path)
        # Convert to grayscale if not already
        image = image.convert('L')
        image_array = np.array(image)

        # Version 1: Blackout top 5% most intense pixels
        top5_threshold = np.percentile(image_array, 95)
        mask_top5 = image_array >= top5_threshold
        image_array_top5_perturbed = image_array.copy()
        image_array_top5_perturbed[mask_top5] = 0

        # Version 2: Blackout bottom 5% least intense (non-zero) pixels
        non_zero_pixels = image_array[image_array > 0]
        if non_zero_pixels.size > 0:
            bottom5_threshold = np.percentile(non_zero_pixels, 5)
            mask_bottom5 = (image_array > 0) & (image_array <= bottom5_threshold)
            image_array_bottom5_perturbed = image_array.copy()
            image_array_bottom5_perturbed[mask_bottom5] = 0
        else:
            image_array_bottom5_perturbed = image_array.copy()

        # Save Version 1
        filename_top5 = os.path.splitext(os.path.basename(image_path))[0] + '_top5_blackout.png'
        output_path_top5 = os.path.join(output_subdir, filename_top5)
        Image.fromarray(image_array_top5_perturbed).save(output_path_top5)

        # Save Version 2
        filename_bottom5 = os.path.splitext(os.path.basename(image_path))[0] + '_bottom5_blackout.png'
        output_path_bottom5 = os.path.join(output_subdir, filename_bottom5)
        Image.fromarray(image_array_bottom5_perturbed).save(output_path_bottom5)


if __name__ == '__main__':
    main()
