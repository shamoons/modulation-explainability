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
        # Scale SNR values to [0, 1] range for better numerical stability
        self.register_buffer('snr_values', torch.tensor(snr_values, dtype=torch.float32))
        self.snr_min = min(snr_values)
        self.snr_max = max(snr_values)
        
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
        
        # Scale SNR values to [0, 1] range
        scaled_true = (true_snr_values - self.snr_min) / (self.snr_max - self.snr_min)
        scaled_pred = (expected_snr - self.snr_min) / (self.snr_max - self.snr_min)
        
        # Compute absolute difference between expected and true SNR
        abs_diff = torch.abs(scaled_pred - scaled_true)
        
        # Weight the cross-entropy loss by the absolute difference
        # This means predictions that are further from the true value get higher loss
        weights = 1.0 + abs_diff
        
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Apply weights to the cross-entropy loss
        weighted_loss = weights * ce_loss
        
        return weighted_loss.mean()

class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_tasks):
        super(DynamicWeightedLoss, self).__init__()
        self.num_tasks = num_tasks
        # Initialize weights to be equal
        self.weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        
    def forward(self, losses):
        # Convert losses to tensor if they aren't already and ensure they're on the same device as weights
        losses = [torch.tensor(l, device=self.weights.device) if not isinstance(l, torch.Tensor) else l.to(self.weights.device) for l in losses]
        
        # Update weights based on loss magnitudes
        # Higher loss = lower weight
        loss_magnitudes = torch.tensor([l.item() for l in losses], device=self.weights.device)
        # Add small epsilon to avoid division by zero
        loss_magnitudes = loss_magnitudes + 1e-8
        # Inverse of loss magnitudes
        new_weights = 1.0 / loss_magnitudes
        # Normalize weights
        new_weights = new_weights / new_weights.sum()
        
        # Update weights directly
        with torch.no_grad():
            self.weights.data = new_weights
        
        # Compute weighted loss
        return sum(w * l for w, l in zip(self.weights, losses))
    
    def get_weights(self):
        return self.weights.detach().cpu().numpy() 