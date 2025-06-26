# src/calculate_pid.py

import os
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def calculate_area_difference(original_path, perturbed_path):
    """
    Calculate the proportion of area changed between the original and perturbed images.
    """
    original = Image.open(original_path).convert("L")
    perturbed = Image.open(perturbed_path).convert("L")

    original_array = np.array(original)
    perturbed_array = np.array(perturbed)

    diff = np.abs(original_array - perturbed_array)
    changed_area = np.sum(diff > 0)
    total_area = original_array.size

    return changed_area / total_area


def calculate_average_changes_by_perturbation(base_dir, modulation_folders, snr_folders, perturbation_mapping):
    """
    Calculate the average area changed for each perturbation type, grouped by Modulation and SNR.
    """
    results = defaultdict(list)

    for perturbation_level, perturbation_name in perturbation_mapping.items():
        print(f"Processing Perturbation: {perturbation_name}")
        for modulation in modulation_folders:
            for snr in snr_folders:
                original_dir = os.path.join(base_dir, f"constellation_diagrams/{modulation}/{snr}/")

                if not os.path.exists(original_dir):
                    print(f"Original directory not found: {original_dir}")
                    continue

                original_images = [f for f in os.listdir(original_dir) if f.startswith(f"grayscale_{modulation}_{snr}_sample_")]

                perturbed_dir = os.path.join(base_dir, f"perturbed_constellations/{modulation}/{snr}/")
                if not os.path.exists(perturbed_dir):
                    print(f"Perturbed directory not found: {perturbed_dir}")
                    continue

                progress = tqdm(original_images, desc=f"Processing {modulation}_{snr}", leave=False)
                local_areas = []
                for original_image in progress:
                    sample_id = original_image.split("_sample_")[-1].split(".png")[0]
                    original_image_path = os.path.join(original_dir, original_image)

                    perturbed_image_name = f"grayscale_{modulation}_{snr}_sample_{sample_id}_{perturbation_level}_blackout.png"
                    perturbed_image_path = os.path.join(perturbed_dir, perturbed_image_name)

                    if os.path.exists(perturbed_image_path):
                        changed_area = calculate_area_difference(original_image_path, perturbed_image_path)
                        local_areas.append(changed_area)
                        progress.set_postfix({'Sample': sample_id, 'Area Changed': f"{changed_area:.4f}"})
                    else:
                        progress.set_postfix({'Missing': perturbed_image_name})

                # Print results for the current modulation and SNR
                if local_areas:
                    avg_changed_area = sum(local_areas) / len(local_areas)
                    print(f"Results for {modulation}, {snr}, Perturbation: {perturbation_name}")
                    print(f"  Average Changed Area: {avg_changed_area:.4f}")
                    results[(perturbation_name, modulation, snr)] = avg_changed_area

    return results


if __name__ == "__main__":
    base_dir = ""  # Specify the base directory path
    modulation_folders = os.listdir(os.path.join(base_dir, "constellation_diagrams"))
    snr_folders = [f"SNR_{i}" for i in range(-20, 30, 2)]

    perturbation_mapping = {
        "top5": "5% Brightest",
        "top2": "2% Brightest",
        "top10": "10% Brightest",
        "top1": "1% Brightest",
        "bottom5": "5% Dimmest",
        "bottom2": "2% Dimmest",
        "bottom1": "1% Dimmest",
    }

    averages = calculate_average_changes_by_perturbation(base_dir, modulation_folders, snr_folders, perturbation_mapping)

    output_file = "average_area_changes_by_perturbation.csv"
    with open(output_file, "w") as f:
        f.write("Perturbation,Modulation,SNR,Average_Changed_Area\n")
        for (perturbation, modulation, snr), avg_changed_area in sorted(averages.items()):
            f.write(f"{perturbation},{modulation},{snr},{avg_changed_area:.4f}\n")

    print(f"Averages grouped by perturbation written to {output_file}")