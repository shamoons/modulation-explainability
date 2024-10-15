# src/utils/image_utils.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import pandas as pd  # Import pandas


def plot_confusion_matrix(true_labels, pred_labels, label_type, epoch, label_names=None, mod_labels=None, snr_labels=None):
    """
    Plot and save a confusion matrix with multi-level labels for 'Combined' type.

    Args:
        true_labels (list of int): True class labels.
        pred_labels (list of int): Predicted class labels.
        label_type (str): Type of label ('Modulation', 'SNR', 'Combined').
        epoch (int): Current epoch number.
        label_names (list of str or None): List of label names corresponding to class indices.
        mod_labels (list of str or None): List of modulation labels for multi-level indexing.
        snr_labels (list of str or None): List of SNR labels for multi-level indexing.
    """
    # Create directory for confusion matrices if it doesn't exist
    save_dir = "confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(true_labels, pred_labels)

    # Adjust figure size based on the size of the confusion matrix
    if label_type == 'Combined':
        figsize = (max(12, cm.shape[1] * 0.6), max(10, cm.shape[0] * 0.6))
    else:
        figsize = (10, 8)

    plt.figure(figsize=figsize)

    if label_type == 'Combined' and mod_labels is not None and snr_labels is not None:
        # Create MultiIndex for the axes
        index = pd.MultiIndex.from_arrays([mod_labels, snr_labels], names=['Modulation', 'SNR'])
        df_cm = pd.DataFrame(cm, index=index, columns=index)

        # Plot heatmap
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")

        plt.xlabel(f"Predicted {label_type} Labels")
        plt.ylabel(f"True {label_type} Labels")
        plt.title(f"{label_type} Confusion Matrix - Epoch {epoch + 1}")

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        # Check if label names are provided, otherwise use numeric labels
        if label_names is None:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xticks([])
            plt.yticks([])
        else:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)

        plt.xlabel(f"Predicted {label_type} Labels")
        plt.ylabel(f"True {label_type} Labels")
        plt.title(f"{label_type} Confusion Matrix - Epoch {epoch + 1}")

    plt.tight_layout()

    # Save confusion matrix
    file_path = os.path.join(save_dir, f"confusion_matrix_{label_type}_epoch_{epoch + 1}.png")
    plt.savefig(file_path)
    plt.close()

    # Log to Weights and Biases
    wandb.log({f"Confusion Matrix {label_type} Epoch {epoch + 1}": wandb.Image(file_path)})
