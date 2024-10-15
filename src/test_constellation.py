# src/test_constellation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from models.constellation_model import ConstellationResNet
from constellation_loader import get_constellation_dataloader
from utils.device_utils import get_device


def validate_across_snrs(model, device, criterion, dataloader, snr_list):
    """
    Validate the model across specific SNRs and return performance metrics.
    """
    results = {snr: {"accuracy": 0, "loss": 0, "total": 0} for snr in snr_list}

    model.eval()
    with torch.no_grad():
        for inputs, labels, snrs in tqdm(dataloader, desc="Validating across SNRs"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            total = labels.size(0)

            for snr in snr_list:
                results[snr]["accuracy"] += correct / total
                results[snr]["loss"] += loss.item() / total
                results[snr]["total"] += 1

    for snr in snr_list:
        if results[snr]["total"] > 0:
            results[snr]["accuracy"] /= results[snr]["total"]
            results[snr]["loss"] /= results[snr]["total"]

    return results


def validate_per_modulation(model, device, criterion, dataloader, mod_list):
    """
    Validate the model across individual modulation types.
    """
    results = {mod: {"accuracy": 0, "loss": 0, "total": 0} for mod in mod_list}

    model.eval()
    with torch.no_grad():
        for inputs, labels, snrs in tqdm(dataloader, desc="Validating across Modulation Types"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            total = labels.size(0)

            for mod in mod_list:
                results[mod]["accuracy"] += correct / total
                results[mod]["loss"] += loss.item() / total
                results[mod]["total"] += 1

    for mod in mod_list:
        if results[mod]["total"] > 0:
            results[mod]["accuracy"] /= results[mod]["total"]
            results[mod]["loss"] /= results[mod]["total"]

    return results


def validate_per_snr(model, device, criterion, dataloader, snr_list):
    """
    Validate the model across individual SNRs and generate results.
    """
    for snr in snr_list:
        print(f"Validating for SNR: {snr}")
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels, snrs in tqdm(dataloader, desc=f"Validating SNR {snr}"):
                # Filter for the current SNR
                inputs = inputs[snrs == snr]
                labels = labels[snrs == snr]
                if len(labels) == 0:
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

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
    batch_size = 512
    snr_list = [0, 20, 30]  # Validate on these SNRs
    mod_list = ['QPSK', 'BPSK', '16QAM', '256QAM']  # Example modulation schemes

    # Load model and validation dataset
    model = ConstellationResNet(num_classes=24, input_channels=1 if image_type == 'grayscale' else 3)

    # Load the best model
    best_model_path = "checkpoints/best_model_epoch_8.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded model from {best_model_path}")

    val_loader = get_constellation_dataloader(root_dir, snr_list=None, image_type=image_type, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    # Get device
    device = get_device()
    model.to(device)

    # Validate across all SNRs
    validate_across_snrs(model, device, criterion, val_loader, snr_list)

    # Validate across modulation types
    validate_per_modulation(model, device, criterion, val_loader, mod_list)

    # Validate for each SNR separately
    validate_per_snr(model, device, criterion, val_loader, snr_list)
