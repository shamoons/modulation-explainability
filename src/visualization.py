# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_time_domain_signals(X, idx=0, mod_type=None, save_path=None):
    """
    Plots the in-phase (I) and quadrature (Q) components of a signal.

    Args:
        X (ndarray): The signal data.
        idx (int): The index of the signal to plot.
        mod_type (str): The modulation type of the signal (for labeling purposes).
        save_path (str): The file path to save the image. If None, the plot is displayed.
    """
    signal = X[idx]

    # in_phase (real part) and quadrature (imaginary part)
    in_phase = signal[:, 0]  # In-phase
    quadrature = signal[:, 1]  # Quadrature

    plt.figure(figsize=(10, 5))

    # Dynamically setting y-axis limits
    plt.subplot(2, 1, 1)
    plt.plot(in_phase)
    plt.title(f"Time-Domain Plot of In-Phase Component - {mod_type}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.ylim(np.min(in_phase) - 0.1, np.max(in_phase) + 0.1)  # Set dynamic limits

    plt.subplot(2, 1, 2)
    plt.plot(quadrature)
    plt.title(f"Time-Domain Plot of Quadrature Component - {mod_type}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.ylim(np.min(quadrature) - 0.1, np.max(quadrature) + 0.1)  # Set dynamic limits

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot after saving
    else:
        plt.show()


def plot_constellation(X, idx=0, mod_type=None, save_path=None):
    """
    Plots the constellation diagram (In-phase vs Quadrature) of a signal.

    Args:
        X (ndarray): The signal data.
        idx (int): The index of the signal to plot.
        mod_type (str): The modulation type of the signal (for labeling purposes).
        save_path (str): The file path to save the image. If None, the plot is displayed.
    """
    signal = X[idx]

    # in_phase (real part) and quadrature (imaginary part)
    in_phase = signal[:, 0]  # In-phase
    quadrature = signal[:, 1]  # Quadrature

    plt.figure(figsize=(6, 6))
    plt.scatter(in_phase, quadrature, s=5, color='blue')
    plt.title(f"Constellation Diagram - {mod_type}")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot after saving
    else:
        plt.show()


# Function to loop through multiple signals and save the plots
def save_all_plots(X, mod_type, save_dir='output/', num_samples=10):
    """
    Saves time-domain and constellation plots for multiple signals.

    Args:
        X (ndarray): The signal data.
        mod_type (str): The modulation type of the signal.
        save_dir (str): The directory to save the images.
        num_samples (int): Number of samples to visualize and save.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_samples):
        time_domain_path = os.path.join(save_dir, f"time_domain_signal_{mod_type}_idx_{i}.png")
        constellation_path = os.path.join(save_dir, f"constellation_{mod_type}_idx_{i}.png")

        plot_time_domain_signals(X, idx=i, mod_type=mod_type, save_path=time_domain_path)
        plot_constellation(X, idx=i, mod_type=mod_type, save_path=constellation_path)
        print(f"Saved plots for sample {i} to {save_dir}")
