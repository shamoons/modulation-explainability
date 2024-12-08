# src/test_constellation.py

import torch
from torch.utils.data import DataLoader
import argparse
import logging
from models.constellation_model import ConstellationResNet  # or your VisionTransformer
from loaders.constellation_loader import ConstellationDataset
from loaders.perturbation_loader import PerturbationDataset
from utils.device_utils import get_device
from tqdm import tqdm  # Import tqdm for progress bars

logging.basicConfig(level=logging.INFO)


def test_model(model, dataloader, criterion, device, description="Testing"):
    model.eval()
    correct_mod = 0
    correct_snr = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        # Wrap the dataloader with tqdm
        for images, mod_labels, snr_labels in tqdm(dataloader, desc=description):
            images, mod_labels, snr_labels = images.to(device), mod_labels.to(device), snr_labels.to(device)

            mod_outputs, snr_outputs = model(images)
            mod_loss = criterion(mod_outputs, mod_labels)
            snr_loss = criterion(snr_outputs, snr_labels)
            loss = mod_loss + snr_loss

            total_loss += loss.item() * images.size(0)  # Multiply by batch size
            _, mod_predicted = torch.max(mod_outputs, 1)
            _, snr_predicted = torch.max(snr_outputs, 1)

            correct_mod += (mod_predicted == mod_labels).sum().item()
            correct_snr += (snr_predicted == snr_labels).sum().item()
            total += mod_labels.size(0)

    avg_loss = total_loss / total
    mod_accuracy = 100 * correct_mod / total
    snr_accuracy = 100 * correct_snr / total
    logging.info(f"Loss: {avg_loss:.4f}, Modulation Accuracy: {mod_accuracy:.2f}%, SNR Accuracy: {snr_accuracy:.2f}%")
    return mod_accuracy, snr_accuracy


def main(args):
    device = get_device()

    # Load model and criterion
    model = ConstellationResNet(num_classes=args.num_mod_classes, snr_classes=args.num_snr_classes, model_name="resnet34")
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Test on non-perturbed data
    logging.info("Testing on non-perturbed data")
    non_perturbed_dataset = ConstellationDataset(
        root_dir=args.data_dir,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process
    )
    non_perturbed_loader = DataLoader(
        non_perturbed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_model(model, non_perturbed_loader, criterion, device, description="Testing unperturbed data")

    # Test on perturbed data (top 5% brightest pixels blacked out)
    logging.info("Testing on perturbed data (top 5% brightest pixels blacked out)")
    perturb_brightest_dataset = PerturbationDataset(
        root_dir=None,  # Not used in the modified PerturbationDataset
        perturbation_dir=args.perturbation_dir,
        perturbation_type='top5_blackout',
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process
    )
    perturb_brightest_loader = DataLoader(
        perturb_brightest_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_model(model, perturb_brightest_loader, criterion, device, description="Testing top 5% blackout")

    # Test on perturbed data (bottom 5% dimmest non-zero pixels blacked out)
    logging.info("Testing on perturbed data (bottom 5% dimmest non-zero pixels blacked out)")
    perturb_dimmest_dataset = PerturbationDataset(
        root_dir=None,  # Not used in the modified PerturbationDataset
        perturbation_dir=args.perturbation_dir,
        perturbation_type='bottom5_blackout',
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process
    )
    perturb_dimmest_loader = DataLoader(
        perturb_dimmest_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_model(model, perturb_dimmest_loader, criterion, device, description="Testing bottom 5% blackout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test constellation model on perturbed and non-perturbed data")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the unperturbed dataset")
    parser.add_argument('--perturbation_dir', type=str, required=True, help="Root directory of the perturbed dataset")
    parser.add_argument('--image_type', type=str, default='grayscale', help="Type of images ('grayscale' or 'RGB')")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    parser.add_argument('--snr_list', type=str, default=None, help="Comma-separated list of SNR values to load")
    parser.add_argument('--mods_to_process', type=str, default=None, help="Comma-separated list of modulation types to load")
    parser.add_argument('--num_mod_classes', type=int, default=10, help="Number of modulation classes")
    parser.add_argument('--num_snr_classes', type=int, default=20, help="Number of SNR classes")

    args = parser.parse_args()

    # Convert comma-separated strings to lists
    if args.snr_list is not None:
        args.snr_list = [int(snr) for snr in args.snr_list.split(',')]
    if args.mods_to_process is not None:
        args.mods_to_process = args.mods_to_process.split(',')

    main(args)
