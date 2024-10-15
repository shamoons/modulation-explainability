# src/test_constellation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from models.constellation_model import ConstellationResNet
from constellation_loader import ConstellationDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.device_utils import get_device
from utils.config_utils import load_loss_config  # Add this to load alpha/beta


def validate_across_snrs(model, device, criterion_modulation, criterion_snr, dataloader, snr_list):
    """
    Validate the model across specific SNRs and return performance metrics.
    Includes multi-task loss using alpha and beta to weigh modulation and SNR losses.
    """
    results = {snr: {"accuracy": 0, "loss": 0, "total": 0} for snr in snr_list}

    # Load alpha and beta weights
    alpha, beta = load_loss_config()

    model.eval()
    with torch.no_grad():
        for inputs, modulation_labels, snr_labels in tqdm(dataloader, desc="Validating across SNRs"):
            inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)
            modulation_output, snr_output = model(inputs)

            # Compute modulation and SNR losses
            loss_modulation = criterion_modulation(modulation_output, modulation_labels)
            loss_snr = criterion_snr(snr_output, snr_labels)

            # Combine losses using alpha and beta
            total_loss = alpha * loss_modulation + beta * loss_snr

            _, predicted_modulation = modulation_output.max(1)
            correct = predicted_modulation.eq(modulation_labels).sum().item()
            total = modulation_labels.size(0)

            for snr in snr_list:
                results[snr]["accuracy"] += correct / total
                results[snr]["loss"] += total_loss.item() / total
                results[snr]["total"] += 1

    for snr in snr_list:
        if results[snr]["total"] > 0:
            results[snr]["accuracy"] /= results[snr]["total"]
            results[snr]["loss"] /= results[snr]["total"]

    return results


def validate_per_modulation(model, device, criterion_modulation, criterion_snr, dataloader, mod_list):
    """
    Validate the model across individual modulation types with multi-task loss.
    """
    results = {mod: {"accuracy": 0, "loss": 0, "total": 0} for mod in mod_list}

    # Load alpha and beta weights
    alpha, beta = load_loss_config()

    model.eval()
    with torch.no_grad():
        for inputs, modulation_labels, snr_labels in tqdm(dataloader, desc="Validating across Modulation Types"):
            inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)
            modulation_output, snr_output = model(inputs)

            # Compute losses for modulation and SNR
            loss_modulation = criterion_modulation(modulation_output, modulation_labels)
            loss_snr = criterion_snr(snr_output, snr_labels)

            total_loss = alpha * loss_modulation + beta * loss_snr

            _, predicted_modulation = modulation_output.max(1)
            correct = predicted_modulation.eq(modulation_labels).sum().item()
            total = modulation_labels.size(0)

            for mod in mod_list:
                results[mod]["accuracy"] += correct / total
                results[mod]["loss"] += total_loss.item() / total
                results[mod]["total"] += 1

    for mod in mod_list:
        if results[mod]["total"] > 0:
            results[mod]["accuracy"] /= results[mod]["total"]
            results[mod]["loss"] /= results[mod]["total"]

    return results


def validate_per_snr(model, device, criterion_modulation, criterion_snr, dataloader, snr_list):
    """
    Validate the model across individual SNRs and generate results with multi-task loss.
    """
    for snr in snr_list:
        print(f"Validating for SNR: {snr}")
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        # Load alpha and beta weights
        alpha, beta = load_loss_config()

        with torch.no_grad():
            for inputs, modulation_labels, snr_labels, snrs in tqdm(dataloader, desc=f"Validating SNR {snr}"):
                inputs = inputs[snrs == snr]
                modulation_labels = modulation_labels[snrs == snr]
                snr_labels = snr_labels[snrs == snr]
                if len(modulation_labels) == 0:
                    continue

                inputs, modulation_labels, snr_labels = inputs.to(device), modulation_labels.to(device), snr_labels.to(device)
                modulation_output, snr_output = model(inputs)

                loss_modulation = criterion_modulation(modulation_output, modulation_labels)
                loss_snr = criterion_snr(snr_output, snr_labels)
                total_loss += alpha * loss_modulation + beta * loss_snr

                _, predicted_modulation = modulation_output.max(1)
                correct += predicted_modulation.eq(modulation_labels).sum().item()
                total += modulation_labels.size(0)

                all_predictions.extend(predicted_modulation.cpu().numpy())
                all_targets.extend(modulation_labels.cpu().numpy())

        total_accuracy = 100.0 * correct / total if total > 0 else 0
        total_loss = total_loss / len(dataloader)

        print(f"SNR: {snr}, Accuracy: {total_accuracy:.2f}%, Loss: {total_loss:.4f}")
        plot_confusion_matrix(all_targets, all_predictions, labels=dataloader.dataset.modulation_labels, snr=snr)


def plot_confusion_matrix(targets, predictions, labels, snr=None):
    """
    Plot the confusion matrix.
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    title = f"Confusion Matrix (SNR: {snr})" if snr else "Confusion Matrix (All SNRs)"
    plt.title(title)
    filename = f"confusion_matrix_snr_{snr}.png" if snr else "confusion_matrix_all_snrs.png"
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # Parameters
    image_type = 'grayscale'
    root_dir = "constellation"
    batch_size = 256
    snr_list = [0, 20, 30]  # Validate on these SNRs
    mod_list = ['QPSK', 'BPSK', '16QAM', '256QAM']  # Example modulation schemes

    # Load dataset
    dataset = ConstellationDataset(root_dir=root_dir, image_type=image_type, snr_list=snr_list)

    # Get the same train/validation split indices used during training
    indices = list(range(len(dataset)))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Use the validation indices for testing
    val_sampler = SubsetRandomSampler(val_idx)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12, pin_memory=True)

    # Load model
    model = ConstellationResNet(num_classes=13, snr_classes=6, input_channels=1 if image_type == 'grayscale' else 3)

    # Load the best model
    best_model_path = "checkpoints/best_model_epoch_1.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded model from {best_model_path}")

    criterion_modulation = torch.nn.CrossEntropyLoss()
    criterion_snr = torch.nn.CrossEntropyLoss()

    # Get device
    device = get_device()
    model.to(device)

    # Validate across all SNRs
    validate_across_snrs(model, device, criterion_modulation, criterion_snr, val_loader, snr_list)

    # Validate across modulation types
    validate_per_modulation(model, device, criterion_modulation, criterion_snr, val_loader, mod_list)

    # Validate for each SNR separately
    validate_per_snr(model, device, criterion_modulation, criterion_snr, val_loader, snr_list)
