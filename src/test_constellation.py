# src/test_constellation.py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm
from models.constellation_model import ConstellationResNet
from loaders.constellation_loader import ConstellationDataset
from utils.device_utils import get_device
from validate_constellation import validate
from utils.image_utils import plot_confusion_matrix, plot_f1_scores
from sklearn.metrics import classification_report
import os


def validate_across_conditions(model, device, criterion_modulation, criterion_snr, val_loader, snr_list, mod_list):
    """
    Validate across all conditions: overall, by SNR, and by modulation type.
    Output confusion matrices and F1 scores in the 'test' directory.
    """
    # Create a 'test' directory for outputs
    os.makedirs('test', exist_ok=True)

    # Overall validation
    print("Validating across all SNRs and Modulation types")
    val_results = validate(model, device, criterion_modulation, criterion_snr, val_loader)

    # Unpack the new values (modulation loss and snr loss)
    val_loss, modulation_loss, snr_loss, val_mod_accuracy, val_snr_accuracy, val_combined_accuracy, all_true_mod_labels, all_pred_mod_labels, all_true_snr_labels, all_pred_snr_labels = val_results

    print(f"Overall Validation Loss: {val_loss}")
    print(f"Modulation Loss: {modulation_loss}")
    print(f"SNR Loss: {snr_loss}")

    # Plot confusion matrices and F1 scores for overall results
    plot_confusion_matrix(
        all_true_mod_labels,
        all_pred_mod_labels,
        label_type='Modulation',
        epoch=0,
        label_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()],
        output_dir='test'
    )
    plot_confusion_matrix(
        all_true_snr_labels,
        all_pred_snr_labels,
        label_type='SNR',
        epoch=0,
        label_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()],
        output_dir='test'
    )

    # F1 scores
    print("\nOverall F1 Scores (Modulation)")
    report_modulation = classification_report(all_true_mod_labels, all_pred_mod_labels, target_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()])
    print(report_modulation)
    with open('test/f1_modulation_report.txt', 'w') as f:
        f.write(report_modulation)

    print("\nOverall F1 Scores (SNR)")
    report_snr = classification_report(all_true_snr_labels, all_pred_snr_labels, target_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()])
    print(report_snr)
    with open('test/f1_snr_report.txt', 'w') as f:
        f.write(report_snr)

    # Validation per SNR
    for snr in snr_list:
        print(f"\nValidating for SNR: {snr}")
        snr_idx = [i for i, (_, _, snr_label) in enumerate(val_loader.dataset) if snr_label == val_loader.dataset.snr_labels[snr]]
        snr_subset = Subset(val_loader.dataset, snr_idx)
        snr_loader = DataLoader(snr_subset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, pin_memory=True)
        val_results_snr = validate(model, device, criterion_modulation, criterion_snr, snr_loader)
        _, _, _, _, _, _, all_true_mod_labels_snr, all_pred_mod_labels_snr = val_results_snr

        # Plot confusion matrix and F1 scores for this SNR
        plot_confusion_matrix(
            all_true_mod_labels_snr,
            all_pred_mod_labels_snr,
            label_type=f'Modulation_SNR_{snr}',
            epoch=0,
            label_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()],
            output_dir='test'
        )

        print(f"\nF1 Scores for SNR: {snr}")
        report_snr_mod = classification_report(all_true_mod_labels_snr, all_pred_mod_labels_snr, target_names=[label for label in val_loader.dataset.inverse_modulation_labels.values()])
        print(report_snr_mod)
        with open(f'test/f1_modulation_snr_{snr}_report.txt', 'w') as f:
            f.write(report_snr_mod)

    # Validation per modulation type
    for mod in mod_list:
        print(f"\nValidating for Modulation Type: {mod}")
        mod_idx = [i for i, (_, mod_label, _) in enumerate(val_loader.dataset) if mod_label == val_loader.dataset.modulation_labels[mod]]
        mod_subset = Subset(val_loader.dataset, mod_idx)
        mod_loader = DataLoader(mod_subset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, pin_memory=True)
        val_results_mod = validate(model, device, criterion_modulation, criterion_snr, mod_loader)
        _, _, _, _, _, _, all_true_snr_labels_mod, all_pred_snr_labels_mod = val_results_mod

        # Plot confusion matrix and F1 scores for this modulation
        plot_confusion_matrix(
            all_true_snr_labels_mod,
            all_pred_snr_labels_mod,
            label_type=f'SNR_Modulation_{mod}',
            epoch=0,
            label_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()],
            output_dir='test'
        )

        print(f"\nF1 Scores for Modulation Type: {mod}")
        report_mod_snr = classification_report(all_true_snr_labels_mod, all_pred_snr_labels_mod, target_names=[str(label) for label in val_loader.dataset.inverse_snr_labels.values()])
        print(report_mod_snr)
        with open(f'test/f1_snr_modulation_{mod}_report.txt', 'w') as f:
            f.write(report_mod_snr)


if __name__ == "__main__":
    # Parameters
    image_type = 'grayscale'
    root_dir = "constellation"
    batch_size = 256
    snr_list = [-10, 0, 6, 10, 20, 30]
    mod_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']

    # Load dataset
    dataset = ConstellationDataset(root_dir=root_dir, image_type=image_type, snr_list=snr_list)

    # Ensure consistent train/validation split
    indices = list(range(len(dataset)))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_sampler = SubsetRandomSampler(val_idx)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12, pin_memory=True)

    # Load model
    model = ConstellationResNet(num_classes=len(mod_list), snr_classes=len(snr_list), input_channels=1 if image_type == 'grayscale' else 3)
    model.load_state_dict(torch.load("checkpoints/best_model_epoch_1.pth"))
    device = get_device()
    model.to(device)

    criterion_modulation = torch.nn.CrossEntropyLoss()
    criterion_snr = torch.nn.CrossEntropyLoss()

    # Validate across all SNRs and Modulation Types
    validate_across_conditions(model, device, criterion_modulation, criterion_snr, val_loader, snr_list, mod_list)
