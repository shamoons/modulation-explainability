#!/usr/bin/env python3
"""
Generate confusion matrices and F1 bar plots from canonical test results.

Inputs (expected):
  papers/ELSP_Paper/results/raw_data/
    - test_labels_modulation.npy
    - test_predictions_modulation.npy
    - test_labels_snr.npy
    - test_predictions_snr.npy
  papers/ELSP_Paper/results/performance_metrics/test_set_results.json

Outputs:
  papers/ELSP_Paper/results/confusion_matrices/
    - modulation_confusion_matrix.png
    - snr_confusion_matrix.png
  papers/ELSP_Paper/results/f1_scores/
    - f1_modulation_bar.png
    - f1_snr_bar.png
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW = RESULTS_DIR / "raw_data"
PERF = RESULTS_DIR / "performance_metrics"
CM_DIR = RESULTS_DIR / "confusion_matrices"
F1_DIR = RESULTS_DIR / "f1_scores"


def ensure_dirs():
    CM_DIR.mkdir(parents=True, exist_ok=True)
    F1_DIR.mkdir(parents=True, exist_ok=True)


def load_arrays():
    y_mod = np.load(RAW / "test_labels_modulation.npy")
    yhat_mod = np.load(RAW / "test_predictions_modulation.npy")
    y_snr = np.load(RAW / "test_labels_snr.npy")
    yhat_snr = np.load(RAW / "test_predictions_snr.npy")
    return y_mod, yhat_mod, y_snr, yhat_snr


def load_label_names():
    names_mod, names_snr = None, None
    tj = PERF / "test_set_results.json"
    if tj.exists():
        with tj.open() as f:
            js = json.load(f)
        mod_map = js.get("f1_scores", {}).get("modulation", {}).get("per_class_f1", {})
        snr_map = js.get("f1_scores", {}).get("snr", {}).get("per_class_f1", {})
        if mod_map:
            names_mod = list(mod_map.keys())
        if snr_map:
            # sort numerically if keys are numeric strings
            try:
                names_snr = [str(x) for x in sorted(map(int, snr_map.keys()))]
            except Exception:
                names_snr = list(snr_map.keys())
    return names_mod, names_snr


def plot_confusion(cm: np.ndarray, title: str, out_path: Path, xticklabels=None, yticklabels=None):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    num_classes = cm.shape[0]
    tick_step = max(1, num_classes // 10)
    ticks = np.arange(0, num_classes, tick_step)
    plt.xticks(ticks, (xticklabels or ticks), rotation=90, fontsize=7)
    plt.yticks(ticks, (yticklabels or ticks), fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_f1_bars(f1s, label_names, title: str, out_path: Path):
    plt.figure(figsize=(10, 4))
    idx = np.arange(len(f1s))
    plt.bar(idx, f1s, color="#4C78A8")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 score")
    plt.title(title)
    if label_names and len(label_names) == len(f1s):
        plt.xticks(idx, label_names, rotation=45, ha="right", fontsize=7)
    else:
        plt.xticks(idx, idx)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_dirs()
    y_mod, yhat_mod, y_snr, yhat_snr = load_arrays()
    names_mod, names_snr = load_label_names()

    # Confusion matrices (normalized per true class)
    cm_mod = confusion_matrix(y_mod, yhat_mod, labels=np.arange(np.max(y_mod) + 1))
    cm_snr = confusion_matrix(y_snr, yhat_snr, labels=np.arange(np.max(y_snr) + 1))
    # Normalize rows for visualization
    cm_mod_norm = cm_mod / (cm_mod.sum(axis=1, keepdims=True) + 1e-9)
    cm_snr_norm = cm_snr / (cm_snr.sum(axis=1, keepdims=True) + 1e-9)
    plot_confusion(cm_mod_norm, "Modulation Confusion (normalized)", CM_DIR / "modulation_confusion_matrix.png")
    plot_confusion(cm_snr_norm, "SNR Confusion (normalized)", CM_DIR / "snr_confusion_matrix.png")

    # F1 scores per class
    f1_mod = f1_score(y_mod, yhat_mod, average=None)
    f1_snr = f1_score(y_snr, yhat_snr, average=None)
    plot_f1_bars(f1_mod, names_mod, "Per-class F1 (Modulation)", F1_DIR / "f1_modulation_bar.png")
    # For SNR labels, format nicely
    if names_snr is None:
        # derive unique labels sorted
        uniq = sorted(set(y_snr.tolist()))
        names_snr = [str(u) for u in uniq]
    plot_f1_bars(f1_snr, names_snr, "Per-class F1 (SNR)", F1_DIR / "f1_snr_bar.png")

    print("Saved:")
    print(" -", CM_DIR / "modulation_confusion_matrix.png")
    print(" -", CM_DIR / "snr_confusion_matrix.png")
    print(" -", F1_DIR / "f1_modulation_bar.png")
    print(" -", F1_DIR / "f1_snr_bar.png")


if __name__ == "__main__":
    main()

