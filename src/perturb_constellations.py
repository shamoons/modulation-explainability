# perturb_constellations.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description='Perturb constellation images by blacking out top and bottom percentage of pixels.')
    parser.add_argument('--percent', type=float, default=5.0, help='Percentage of pixels to blackout (top and bottom). Default is 5.')
    args = parser.parse_args()

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

        # Top X% blackout
        top_threshold = 100 - args.percent
        topX_threshold = np.percentile(image_array, top_threshold)
        mask_topX = image_array >= topX_threshold
        image_array_topX_perturbed = image_array.copy()
        image_array_topX_perturbed[mask_topX] = 0

        # Bottom X% blackout (non-zero pixels)
        non_zero_pixels = image_array[image_array > 0]
        if non_zero_pixels.size > 0:
            bottomX_threshold = np.percentile(non_zero_pixels, args.percent)
            mask_bottomX = (image_array > 0) & (image_array <= bottomX_threshold)
            image_array_bottomX_perturbed = image_array.copy()
            image_array_bottomX_perturbed[mask_bottomX] = 0
        else:
            image_array_bottomX_perturbed = image_array.copy()

        # Save Version 1
        filename_topX = os.path.splitext(os.path.basename(image_path))[0] + f'_top{args.percent}_blackout.png'
        output_path_topX = os.path.join(output_subdir, filename_topX)
        Image.fromarray(image_array_topX_perturbed).save(output_path_topX)

        # Save Version 2
        filename_bottomX = os.path.splitext(os.path.basename(image_path))[0] + f'_bottom{args.percent}_blackout.png'
        output_path_bottomX = os.path.join(output_subdir, filename_bottomX)
        Image.fromarray(image_array_bottomX_perturbed).save(output_path_bottomX)


if __name__ == '__main__':
    main()
