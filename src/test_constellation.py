# src/test_constellation.py

import torch
from torch.utils.data import DataLoader, Subset
import argparse
import logging
from models.constellation_model import ConstellationResNet  # or your VisionTransformer
from loaders.constellation_loader import ConstellationDataset
from loaders.perturbation_loader import PerturbationDataset
from utils.device_utils import get_device
from tqdm import tqdm  # For progress bars
from sklearn.model_selection import train_test_split
from utils.image_utils import plot_confusion_matrix, plot_f1_scores

logging.basicConfig(level=logging.INFO)


def test_model(model, dataloader, criterion, device, description="Testing"):
    model.eval()
    correct_mod = 0
    correct_snr = 0
    total = 0
    total_loss = 0

    all_true_modulation_labels = []
    all_pred_modulation_labels = []
    all_true_snr_labels = []
    all_pred_snr_labels = []

    with torch.no_grad():
        with tqdm(dataloader, desc=description) as progress:
            for images, mod_labels, snr_labels in progress:
                images, mod_labels, snr_labels = images.to(device), mod_labels.to(device), snr_labels.to(device)

                mod_outputs, snr_outputs = model(images)
                mod_loss = criterion(mod_outputs, mod_labels)
                snr_loss = criterion(snr_outputs, snr_labels)
                loss = mod_loss + snr_loss

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                _, mod_predicted = torch.max(mod_outputs, 1)
                _, snr_predicted = torch.max(snr_outputs, 1)

                correct_mod += (mod_predicted == mod_labels).sum().item()
                correct_snr += (snr_predicted == snr_labels).sum().item()
                total += batch_size

                # Store labels for confusion matrix and F1 score
                all_true_modulation_labels.extend(mod_labels.cpu().tolist())
                all_pred_modulation_labels.extend(mod_predicted.cpu().tolist())
                all_true_snr_labels.extend(snr_labels.cpu().tolist())
                all_pred_snr_labels.extend(snr_predicted.cpu().tolist())

                # Update the progress bar with current metrics
                current_loss = (total_loss / total) if total > 0 else 0.0
                current_mod_acc = (100.0 * correct_mod / total) if total > 0 else 0.0
                current_snr_acc = (100.0 * correct_snr / total) if total > 0 else 0.0

                progress.set_postfix({
                    'Loss': f"{current_loss:.4g}",
                    'Mod Acc': f"{current_mod_acc:.2f}%",
                    'SNR Acc': f"{current_snr_acc:.2f}%"
                })

    avg_loss = total_loss / total if total > 0 else 0.0
    mod_accuracy = 100 * correct_mod / total if total > 0 else 0.0
    snr_accuracy = 100 * correct_snr / total if total > 0 else 0.0
    logging.info(f"Loss: {avg_loss:.4f}, Modulation Accuracy: {mod_accuracy:.2f}%, SNR Accuracy: {snr_accuracy:.2f}%")

    return mod_accuracy, snr_accuracy, all_true_modulation_labels, all_pred_modulation_labels, all_true_snr_labels, all_pred_snr_labels


def create_subset(dataset, test_size=0.2, random_state=42):
    """Create a 20% subset of the given dataset for testing."""
    indices = list(range(len(dataset)))
    _, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
    return Subset(dataset, test_idx)


def main(args):
    device = get_device()

    # Load model and criterion
    model = ConstellationResNet(num_classes=args.num_mod_classes, snr_classes=args.num_snr_classes, model_name="resnet18", input_channels=1)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Define SNR categories if using buckets (assuming 3 classes: Low, Medium, High)
    snr_label_names = ["Low", "Medium", "High"]

    # Test on non-perturbed data (20% subset)
    scenario_name_unperturbed = "Unperturbed"
    logging.info(f"Testing on {scenario_name_unperturbed.lower()} data")
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
    mod_acc_np, snr_acc_np, true_mod_np, pred_mod_np, true_snr_np, pred_snr_np = test_model(
        model, non_perturbed_loader, criterion, device, description=f"Testing {scenario_name_unperturbed.lower()} data"
    )

    # For modulation labels, we still rely on dataset's inverse labels
    modulation_label_names = [label for label in non_perturbed_dataset.inverse_modulation_labels.values()]

    # Plot and save confusion matrix and f1 scores for unperturbed
    plot_confusion_matrix(true_mod_np, pred_mod_np, f"{scenario_name_unperturbed} Modulation", 0, label_names=modulation_label_names, output_dir='confusion_matrices')
    plot_confusion_matrix(true_snr_np, pred_snr_np, f"{scenario_name_unperturbed} SNR", 0, label_names=snr_label_names, output_dir='confusion_matrices')
    plot_f1_scores(true_mod_np, pred_mod_np, modulation_label_names, f"{scenario_name_unperturbed} Modulation", 0, output_dir='f1_scores')
    plot_f1_scores(true_snr_np, pred_snr_np, snr_label_names, f"{scenario_name_unperturbed} SNR", 0, output_dir='f1_scores')

    # Test on perturbed data (top 5% brightest pixels blacked out, 20% subset)
    scenario_name_brightest = "Top 5% Brightest"
    logging.info(f"Testing on {scenario_name_brightest.lower()} data")
    perturb_brightest_dataset = PerturbationDataset(
        root_dir=None,
        perturbation_dir=args.perturbation_dir,
        perturbation_type='top5_blackout',
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        use_snr_buckets=True
    )
    perturb_brightest_subset = create_subset(perturb_brightest_dataset, test_size=0.2)
    perturb_brightest_loader = DataLoader(
        perturb_brightest_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    mod_acc_pb, snr_acc_pb, true_mod_pb, pred_mod_pb, true_snr_pb, pred_snr_pb = test_model(
        model, perturb_brightest_loader, criterion, device, description=f"Testing {scenario_name_brightest.lower()} data"
    )

    # Plot and save confusion matrix and f1 scores for top 5% brightest
    plot_confusion_matrix(true_mod_pb, pred_mod_pb, f"{scenario_name_brightest} Modulation", 1, label_names=modulation_label_names, output_dir='confusion_matrices')
    plot_confusion_matrix(true_snr_pb, pred_snr_pb, f"{scenario_name_brightest} SNR", 1, label_names=snr_label_names, output_dir='confusion_matrices')
    plot_f1_scores(true_mod_pb, pred_mod_pb, modulation_label_names, f"{scenario_name_brightest} Modulation", 1, output_dir='f1_scores')
    plot_f1_scores(true_snr_pb, pred_snr_pb, snr_label_names, f"{scenario_name_brightest} SNR", 1, output_dir='f1_scores')

    # Test on perturbed data (bottom 5% dimmest non-zero pixels blacked out, 20% subset)
    scenario_name_dimmest = "Bottom 5% Dimmest"
    logging.info(f"Testing on {scenario_name_dimmest.lower()} data")
    perturb_dimmest_dataset = PerturbationDataset(
        root_dir=None,
        perturbation_dir=args.perturbation_dir,
        perturbation_type='bottom5_blackout',
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        use_snr_buckets=True
    )
    perturb_dimmest_subset = create_subset(perturb_dimmest_dataset, test_size=0.2)
    perturb_dimmest_loader = DataLoader(
        perturb_dimmest_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    mod_acc_pd, snr_acc_pd, true_mod_pd, pred_mod_pd, true_snr_pd, pred_snr_pd = test_model(
        model, perturb_dimmest_loader, criterion, device, description=f"Testing {scenario_name_dimmest.lower()} data"
    )

    # Plot and save confusion matrix and f1 scores for bottom 5% dimmest
    plot_confusion_matrix(true_mod_pd, pred_mod_pd, f"{scenario_name_dimmest} Modulation", 2, label_names=modulation_label_names, output_dir='confusion_matrices')
    plot_confusion_matrix(true_snr_pd, pred_snr_pd, f"{scenario_name_dimmest} SNR", 2, label_names=snr_label_names, output_dir='confusion_matrices')
    plot_f1_scores(true_mod_pd, pred_mod_pd, modulation_label_names, f"{scenario_name_dimmest} Modulation", 2, output_dir='f1_scores')
    plot_f1_scores(true_snr_pd, pred_snr_pd, snr_label_names, f"{scenario_name_dimmest} SNR", 2, output_dir='f1_scores')

    scenario_name_brightest = "Top 1% Brightest"
    logging.info(f"Testing on {scenario_name_brightest.lower()} data")
    perturb_brightest_dataset = PerturbationDataset(
        root_dir=None,
        perturbation_dir=args.perturbation_dir,
        perturbation_type='top1_blackout',
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        use_snr_buckets=True
    )
    perturb_brightest_subset = create_subset(perturb_brightest_dataset, test_size=0.2)
    perturb_brightest_loader = DataLoader(
        perturb_brightest_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    mod_acc_pb, snr_acc_pb, true_mod_pb, pred_mod_pb, true_snr_pb, pred_snr_pb = test_model(
        model, perturb_brightest_loader, criterion, device, description=f"Testing {scenario_name_brightest.lower()} data"
    )

    # Plot and save confusion matrix and f1 scores for top 5% brightest
    plot_confusion_matrix(true_mod_pb, pred_mod_pb, f"{scenario_name_brightest} Modulation", 1, label_names=modulation_label_names, output_dir='confusion_matrices')
    plot_confusion_matrix(true_snr_pb, pred_snr_pb, f"{scenario_name_brightest} SNR", 1, label_names=snr_label_names, output_dir='confusion_matrices')
    plot_f1_scores(true_mod_pb, pred_mod_pb, modulation_label_names, f"{scenario_name_brightest} Modulation", 1, output_dir='f1_scores')
    plot_f1_scores(true_snr_pb, pred_snr_pb, snr_label_names, f"{scenario_name_brightest} SNR", 1, output_dir='f1_scores')

    # Test on perturbed data (bottom 5% dimmest non-zero pixels blacked out, 20% subset)
    scenario_name_dimmest = "Bottom 1% Dimmest"
    logging.info(f"Testing on {scenario_name_dimmest.lower()} data")
    perturb_dimmest_dataset = PerturbationDataset(
        root_dir=None,
        perturbation_dir=args.perturbation_dir,
        perturbation_type='bottom1_blackout',
        image_type=args.image_type,
        snr_list=args.snr_list,
        mods_to_process=args.mods_to_process,
        use_snr_buckets=True
    )
    perturb_dimmest_subset = create_subset(perturb_dimmest_dataset, test_size=0.2)
    perturb_dimmest_loader = DataLoader(
        perturb_dimmest_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    mod_acc_pd, snr_acc_pd, true_mod_pd, pred_mod_pd, true_snr_pd, pred_snr_pd = test_model(
        model, perturb_dimmest_loader, criterion, device, description=f"Testing {scenario_name_dimmest.lower()} data"
    )

    # Plot and save confusion matrix and f1 scores for bottom 5% dimmest
    plot_confusion_matrix(true_mod_pd, pred_mod_pd, f"{scenario_name_dimmest} Modulation", 2, label_names=modulation_label_names, output_dir='confusion_matrices')
    plot_confusion_matrix(true_snr_pd, pred_snr_pd, f"{scenario_name_dimmest} SNR", 2, label_names=snr_label_names, output_dir='confusion_matrices')
    plot_f1_scores(true_mod_pd, pred_mod_pd, modulation_label_names, f"{scenario_name_dimmest} Modulation", 2, output_dir='f1_scores')
    plot_f1_scores(true_snr_pd, pred_snr_pd, snr_label_names, f"{scenario_name_dimmest} SNR", 2, output_dir='f1_scores')


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
