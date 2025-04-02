import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSNRLoss(nn.Module):
    def __init__(self, snr_values):
        """
        Initialize the weighted SNR loss.
        
        Args:
            snr_values (list): List of SNR values in order (e.g., [-20, -18, ..., 30])
        """
        super(WeightedSNRLoss, self).__init__()
        self.register_buffer('snr_values', torch.tensor(snr_values, dtype=torch.float32))
        
    def forward(self, predictions, targets):
        """
        Compute weighted loss based on distance from true SNR.
        
        Args:
            predictions (torch.Tensor): Model predictions (N, num_classes)
            targets (torch.Tensor): True SNR class indices (N,)
            
        Returns:
            torch.Tensor: Weighted loss value
        """
        # Get the predicted probabilities for each SNR class
        probs = F.softmax(predictions, dim=1)
        
        # Get the true SNR values for the targets
        true_snr_values = self.snr_values[targets].to(predictions.device)
        
        # Compute expected SNR value from predictions
        expected_snr = torch.sum(probs * self.snr_values.to(predictions.device), dim=1)
        
        # Compute absolute difference between expected and true SNR
        abs_diff = torch.abs(expected_snr - true_snr_values)
        
        # Weight the cross-entropy loss by the absolute difference
        # This means predictions that are further from the true value get higher loss
        weights = 1.0 + abs_diff
        
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Apply weights to the cross-entropy loss
        weighted_loss = weights * ce_loss
        
        return weighted_loss.mean() 