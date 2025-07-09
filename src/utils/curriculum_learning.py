# src/utils/curriculum_learning.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import wandb
from torch.utils.data import WeightedRandomSampler, DataLoader


class SNRCurriculumScheduler:
    """
    Implements sliding window curriculum learning for SNR classification.
    
    Strategy: Start with all samples from highest SNR (30 dB) and gradually
    include more samples from lower SNRs in a sliding window fashion.
    
    Example progression:
    - Epoch 0: 100% of 30dB, 10% of others
    - Epoch 1: 100% of 30dB, 50% of 28dB, 25% of 26dB, 10% of others
    - Epoch 2: 100% of 30dB, 100% of 28dB, 50% of 26dB, 25% of 24dB, 10% of others
    - etc.
    """
    
    def __init__(
        self, 
        snr_list: List[int],
        min_sample_rate: float = 0.1,  # Minimum sampling rate for SNRs outside window
        window_size: int = 3,  # Number of SNRs in the sliding window
        epochs_per_shift: int = 1  # Epochs before shifting window
    ):
        """
        Args:
            snr_list: Full list of SNR values in the dataset
            min_sample_rate: Minimum sampling rate for SNRs not in focus (default 0.1 = 10%)
            window_size: Number of SNRs to include in sliding window
            epochs_per_shift: How many epochs before shifting the window
        """
        self.full_snr_list = sorted([s for s in snr_list if s >= 0], reverse=True)  # Sort high to low
        self.snr_to_idx = {snr: idx for idx, snr in enumerate(self.full_snr_list)}
        self.min_sample_rate = min_sample_rate
        self.window_size = window_size
        self.epochs_per_shift = epochs_per_shift
        self.current_epoch = 0
        
        # Initialize weights for each SNR
        self.snr_weights = {snr: min_sample_rate for snr in self.full_snr_list}
        self._update_weights()
        
        print(f"Curriculum strategy: Sliding window (high to low)")
        print(f"SNR range: {max(self.full_snr_list)} to {min(self.full_snr_list)} dB")
        print(f"Window size: {window_size} SNRs, shifting every {epochs_per_shift} epochs")
        
    def _update_weights(self):
        """Update sampling weights based on sliding window position."""
        # Calculate window position based on epoch
        window_position = self.current_epoch // self.epochs_per_shift
        
        # Reset all weights to minimum
        for snr in self.full_snr_list:
            self.snr_weights[snr] = self.min_sample_rate
        
        # Apply sliding window weights
        for i, snr in enumerate(self.full_snr_list):
            if i <= window_position:
                # SNRs that have been fully introduced
                self.snr_weights[snr] = 1.0
            elif i <= window_position + self.window_size:
                # SNRs in the current window (gradual introduction)
                position_in_window = i - window_position - 1
                # Weight decreases as we go further into the window
                # First SNR in window: 50%, Second: 25%, Third: 12.5%, etc.
                self.snr_weights[snr] = 0.5 ** (position_in_window + 1)
        
        # Print current focus
        if self.current_epoch % self.epochs_per_shift == 0:
            fully_sampled = [snr for snr in self.full_snr_list if self.snr_weights[snr] == 1.0]
            window_snrs = [(snr, self.snr_weights[snr]) for snr in self.full_snr_list 
                          if 0 < self.snr_weights[snr] < 1.0]
            
            print(f"\nEpoch {self.current_epoch} curriculum focus:")
            print(f"  Fully sampled (100%): {fully_sampled}")
            if window_snrs:
                print(f"  Window (partial sampling):")
                for snr, weight in window_snrs:
                    print(f"    {snr} dB: {weight*100:.1f}%")
    
    def get_sample_weights(self, dataset) -> List[float]:
        """
        Get sampling weights for each sample in the dataset based on SNR.
        
        Args:
            dataset: The dataset containing samples
            
        Returns:
            List of weights for WeightedRandomSampler
        """
        weights = []
        for idx in range(len(dataset)):
            _, _, snr_label = dataset[idx]
            # Get the actual SNR value from the dataset's SNR labels
            snr_value = dataset.inverse_snr_labels[snr_label]
            # Use the weight for this SNR
            weight = self.snr_weights.get(snr_value, self.min_sample_rate)
            weights.append(weight)
        
        return weights
    
    def update_epoch(self, epoch: int):
        """Update current epoch and recalculate weights."""
        self.current_epoch = epoch
        self._update_weights()
        
        # Log weight distribution only if wandb is initialized
        if wandb.run is not None:
            wandb.log({
                f"curriculum/weight_{snr}dB": weight 
                for snr, weight in self.snr_weights.items()
            })
            
            # Log window position
            window_position = epoch // self.epochs_per_shift
            wandb.log({
                "curriculum/window_position": window_position,
                "curriculum/epoch": epoch
            })
    
    def get_distribution_stats(self, indices: List[int], dataset) -> Dict[str, float]:
        """
        Calculate the distribution of SNR values in a set of indices.
        
        Args:
            indices: List of dataset indices
            dataset: The dataset
            
        Returns:
            Dictionary with SNR distribution percentages
        """
        snr_counts = {snr: 0 for snr in self.full_snr_list}
        
        for idx in indices:
            _, _, snr_label = dataset[idx]
            # Get the actual SNR value from the dataset's SNR labels
            snr_value = dataset.inverse_snr_labels[snr_label]
            if snr_value in snr_counts:
                snr_counts[snr_value] += 1
        
        total = len(indices)
        distribution = {f"SNR_{snr}dB": (count / total * 100) if total > 0 else 0 
                       for snr, count in snr_counts.items()}
        
        # Add summary stats
        fully_sampled = sum(snr_counts[s] for s in self.full_snr_list 
                           if self.snr_weights.get(s, 0) == 1.0)
        partial_sampled = sum(snr_counts[s] for s in self.full_snr_list 
                             if 0 < self.snr_weights.get(s, 0) < 1.0)
        minimal_sampled = sum(snr_counts[s] for s in self.full_snr_list 
                             if self.snr_weights.get(s, 0) == self.min_sample_rate)
        
        distribution['Fully_sampled_%'] = (fully_sampled / total * 100) if total > 0 else 0
        distribution['Partial_sampled_%'] = (partial_sampled / total * 100) if total > 0 else 0
        distribution['Minimal_sampled_%'] = (minimal_sampled / total * 100) if total > 0 else 0
        
        return distribution
    
    def print_distribution(self, train_indices: List[int], val_indices: List[int], dataset):
        """Print the SNR distribution for train and validation sets."""
        train_dist = self.get_distribution_stats(train_indices, dataset)
        val_dist = self.get_distribution_stats(val_indices, dataset)
        
        print(f"\nEpoch {self.current_epoch} SNR Distribution:")
        print(f"{'SNR':<8} {'Weight':<8} {'Train %':<10} {'Val %':<10}")
        print("-" * 40)
        
        # Print individual SNRs with their weights
        for snr in self.full_snr_list:
            key = f"SNR_{snr}dB"
            weight = self.snr_weights[snr]
            train_pct = train_dist.get(key, 0)
            val_pct = val_dist.get(key, 0)
            
            # Highlight based on weight
            if weight == 1.0:
                marker = "*"  # Fully sampled
            elif weight > self.min_sample_rate:
                marker = "+"  # In window
            else:
                marker = " "  # Minimal sampling
                
            print(f"{snr:>3} dB{marker}  {weight:>6.1%}   {train_pct:>6.2f}%    {val_pct:>6.2f}%")
        
        print("-" * 40)
        print(f"{'Summary':<16} {'Train %':<10} {'Val %':<10}")
        print(f"{'Fully sampled:':<16} {train_dist['Fully_sampled_%']:>6.2f}%    {val_dist['Fully_sampled_%']:>6.2f}%")
        print(f"{'Partial:':<16} {train_dist['Partial_sampled_%']:>6.2f}%    {val_dist['Partial_sampled_%']:>6.2f}%")
        print(f"{'Minimal:':<16} {train_dist['Minimal_sampled_%']:>6.2f}%    {val_dist['Minimal_sampled_%']:>6.2f}%")
        print(f"\n* = 100% sampling, + = partial sampling (window), blank = {self.min_sample_rate*100:.0f}% sampling")


def create_curriculum_sampler(dataset, indices: List[int], curriculum_scheduler: SNRCurriculumScheduler, 
                            batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create a weighted sampler that adjusts sampling probability based on curriculum.
    
    Args:
        dataset: The full dataset
        indices: Indices for this subset (train/val/test)
        curriculum_scheduler: The curriculum scheduler
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle (should be True for train, False for val/test)
    
    Returns:
        A DataLoader with curriculum-based weighted sampling
    """
    if shuffle and curriculum_scheduler is not None:
        # Get weights for all samples in dataset
        all_weights = curriculum_scheduler.get_sample_weights(dataset)
        
        # Extract weights only for the given indices
        subset_weights = [all_weights[idx] for idx in indices]
        
        # Create a subset dataset that only contains our indices
        from torch.utils.data import Subset
        subset_dataset = Subset(dataset, indices)
        
        # Create weighted sampler for the subset
        sampler = WeightedRandomSampler(
            weights=subset_weights,
            num_samples=len(indices),
            replacement=True
        )
        
        return DataLoader(
            subset_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=4
        )
    else:
        # Standard sampling for validation/test or when curriculum is disabled
        from torch.utils.data.sampler import SubsetRandomSampler
        sampler = SubsetRandomSampler(indices)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=4
        )