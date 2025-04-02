# src/test_constellation.py

import torch
from torch.utils.data import DataLoader, Subset
import argparse
import logging
from models.constellation_model import ConstellationResNet
from loaders.constellation_loader import ConstellationDataset
from loaders.perturbation_loader import PerturbationDataset
from utils.device_utils import get_device
from utils.image_utils import plot_confusion_matrix, plot_f1_scores
from validate_constellation import validate
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(message)s')


def create_subset(dataset, test_size=0.2, random_state=42):
    """Create a subset of the given dataset for testing."""
    indices = list(range(len(dataset)))
    _, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
    return Subset(dataset, test_idx)


def evaluate_and_plot(model, loader, device, criterion_modulation, criterion_snr, scenario_name, epoch):
    """Run validation and plot results for the given data loader."""

    total_samples = len(loader.dataset)
    logging.info(f"Total number of samples in {scenario_name}: {total_samples}")

    val_loss, modulation_loss_total, snr_loss_total, mod_acc, snr_acc, combined_acc, true_mod, pred_mod, true_snr, pred_snr = validate(
        model,
        device,
        criterion_modulation,
        criterion_snr,
        val_loader=loader,  # Explicitly specify the argument
        use_snr_buckets=True,
        use_autocast=False
    )

    if isinstance(loader.dataset, Subset):
        original_dataset = loader.dataset.dataset  # Access the original dataset
    else:
        original_dataset = loader.dataset

    snr_label_names = [str(label) for label in original_dataset.inverse_snr_labels.values()]
    modulation_label_names = [label for label in original_dataset.inverse_modulation_labels.values()]

    # Log results
    logging.info(f"{scenario_name} Results:")
    logging.info(f"  Validation Loss: {val_loss:.4g}")
    logging.info(f"  Modulation Accuracy: {mod_acc:.2f}%")
    logging.info(f"  SNR Accuracy: {snr_acc:.2f}%")
    logging.info(f"  Combined Accuracy: {combined_acc:.2f}%")

    # Plot confusion matrices and F1 scores
    plot_confusion_matrix(
        true_mod,
        pred_mod,
        f"{scenario_name} Modulation",
        epoch,
        label_names=modulation_label_names,
        output_dir='confusion_matrices'
    )
    plot_confusion_matrix(
        true_snr,
        pred_snr,
        f"{scenario_name} SNR",
        epoch,
        label_names=snr_label_names,
        output_dir='confusion_matrices'
    )
    plot_f1_scores(
        true_mod,
        pred_mod,
        label_names=modulation_label_names,
        label_type=f"{scenario_name} Modulation",
        epoch=epoch,
        output_dir='f1_scores'
    )
    plot_f1_scores(
        true_snr,
        pred_snr,
        label_names=snr_label_names,
        label_type=f"{scenario_name} SNR",
        epoch=epoch,
        output_dir='f1_scores'
    )


def main(args):
    device = get_device()

    # Load model and criterion
    model = ConstellationResNet(
        num_classes=args.num_mod_classes,
        snr_classes=args.num_snr_classes,
        model_name="resnet18",
        input_channels=1
    )
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)

    criterion_modulation = torch.nn.CrossEntropyLoss()
    criterion_snr = torch.nn.CrossEntropyLoss()

    # Test on non-perturbed data
    scenario_name_unperturbed = "Unperturbed"
    logging.info(f"Testing {scenario_name_unperturbed} dataset")
    non_perturbed_dataset = ConstellationDataset(
        root_dir=args.data_dir,
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        use_snr_buckets=True
    )
    non_perturbed_subset = create_subset(non_perturbed_dataset, test_size=0.2)
    non_perturbed_loader = DataLoader(
        non_perturbed_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    evaluate_and_plot(
        model,
        non_perturbed_loader,
        device,
        criterion_modulation,
        criterion_snr,
        scenario_name_unperturbed,
        epoch=0
    )

    # Test on perturbed datasets
    perturbations = [
        # ("Top 5% Brightest", "top5_blackout", 1),
        # ("Bottom 5% Dimmest", "bottom5_blackout", 2),
        # ("Top 1% Brightest", "top1_blackout", 3),
        # ("Bottom 1% Dimmest", "bottom1_blackout", 4),
        # ("Top 2% Brightest", "top2_blackout", 5),
        # ("Bottom 2% Dimmest", "bottom2_blackout", 6),
        # ("Top 10% Brightest", "top10_blackout", 5),
        # ("Bottom 10% Dimmest", "bottom10_blackout", 6),
        ("Top 50% Brightest", "top50_blackout", 7),
        ("Bottom 50% Dimmest", "bottom50_blackout", 8),
    ]

    for scenario_name, perturbation_type, epoch in perturbations:
        logging.info(f"\n\nTesting {scenario_name} dataset")
        perturbed_dataset = PerturbationDataset(
            root_dir=None,
            perturbation_dir=args.perturbation_dir,
            perturbation_type=perturbation_type,
            image_type=args.image_type,
            snr_list=args.snr_list,
            mods_to_process=args.mods_to_process,
            use_snr_buckets=True
        )
        perturbed_subset = create_subset(perturbed_dataset, test_size=0.2)
        perturbed_loader = DataLoader(
            perturbed_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        evaluate_and_plot(
            model,
            perturbed_loader,
            device,
            criterion_modulation,
            criterion_snr,
            scenario_name,
            epoch
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test constellation model on perturbed and non-perturbed data")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the unperturbed dataset")
    parser.add_argument('--perturbation_dir', type=str, required=True, help="Root directory of the perturbed dataset")
    parser.add_argument('--image_type', type=str, default='grayscale', help="Type of images ('grayscale' or 'RGB')")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for testing")
    parser.add_argument('--snr_list', type=str, default=None, help="Comma-separated list of SNR values to load")
    parser.add_argument('--mods_to_process', type=str, default=None, help="Comma-separated list of modulation types to load")
    parser.add_argument('--num_mod_classes', type=int, default=20, help="Number of modulation classes")
    parser.add_argument('--num_snr_classes', type=int, default=3, help="Number of SNR classes")

    args = parser.parse_args()

    # Convert comma-separated strings to lists
    if args.snr_list is not None:
        args.snr_list = [int(snr) for snr in args.snr_list.split(',')]
    if args.mods_to_process is not None:
        args.mods_to_process = args.mods_to_process.split(',')

    main(args)
