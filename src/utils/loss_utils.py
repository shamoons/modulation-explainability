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
        # Initialize log variances for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        # Convert losses to tensor if they aren't already and ensure they're on the same device as log_vars
        losses = [torch.tensor(l, device=self.log_vars.device) if not isinstance(l, torch.Tensor) else l.to(self.log_vars.device) for l in losses]
        
        # Kendall's formula: loss = sum(0.5*exp(-s_i)*L_i + 0.5*s_i)
        # where s_i are log variances and L_i are task losses
        weighted_loss = 0
        for i, loss in enumerate(losses):
            # 0.5 * exp(-s_i) * L_i
            weighted_loss += 0.5 * torch.exp(-self.log_vars[i]) * loss
            # 0.5 * s_i
            weighted_loss += 0.5 * self.log_vars[i]
        
        return weighted_loss
    
    def get_weights(self):
        # For monitoring: convert log variances to weights
        # w_i = exp(-s_i) / sum(exp(-s_i))
        weights = torch.exp(-self.log_vars)
        return (weights / weights.sum()).detach().cpu().numpy() 