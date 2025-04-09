import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSNRLoss(nn.Module):
    def __init__(self, snr_values, device):
        """
        Initialize the weighted SNR loss.
        
        Args:
            snr_values (list): List of SNR values in order (e.g., [-20, -18, ..., 30])
            device (torch.device): Device to place tensors on
        """
        super().__init__()
        self.device = device
        
        # Initialize SNR values on the correct device
        self.snr_values = torch.tensor(snr_values, dtype=torch.float32, device=device)
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
        true_snr_values = self.snr_values[targets]
        
        # Compute expected SNR value from predictions
        expected_snr = torch.sum(probs * self.snr_values, dim=1)
        
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

    def scale_to_snr(self, predictions):
        """
        Convert class predictions to actual SNR values.
        For backward compatibility with the regression approach.
        
        Args:
            predictions (torch.Tensor): Predictions from model [batch_size, num_classes]
            
        Returns:
            torch.Tensor: Expected SNR values [batch_size, 1]
        """
        # Get softmax probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Calculate expected SNR value (weighted average)
        expected_snr = torch.sum(probs * self.snr_values, dim=1, keepdim=True)
        
        return expected_snr

    def get_snr_metrics(self, predictions, targets):
        """
        Calculate both classification accuracy and regression metrics (MAE).
        
        Args:
            predictions (torch.Tensor): Predictions from model [batch_size, num_classes]
            targets (torch.Tensor): True SNR class indices [batch_size]
            
        Returns:
            tuple: (accuracy, mae)
            - accuracy: Classification accuracy (percentage)
            - mae: Mean absolute error between expected and true SNR values
        """
        # Get the predicted class (argmax)
        pred_classes = torch.argmax(predictions, dim=1)
        accuracy = (pred_classes == targets).float().mean().item() * 100
        
        # Get the expected SNR value from probabilities
        expected_snr = self.scale_to_snr(predictions).squeeze()
        
        # Get the true SNR values for the targets
        true_snr_values = self.snr_values[targets]
        
        # Calculate MAE
        mae = torch.abs(expected_snr - true_snr_values).mean().item()
        
        return accuracy, mae

class SNRRegressionLoss(nn.Module):
    """
    Loss function for SNR regression with bounded output.
    Handles scaling between [0,1] and [-20,30] ranges.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mse = nn.MSELoss()
        
        # Constants for scaling
        self.MIN_SNR = -20
        self.MAX_SNR = 30
        self.SNR_RANGE = self.MAX_SNR - self.MIN_SNR
        
    def scale_to_01(self, snr):
        """Scale SNR from [-20,30] to [0,1]"""
        return (snr - self.MIN_SNR) / self.SNR_RANGE
    
    def scale_to_snr(self, scaled):
        """Scale from [0,1] to [-20,30]"""
        return scaled * self.SNR_RANGE + self.MIN_SNR
    
    # Add alias for backward compatibility
    def scale_to_snr_range(self, scaled):
        """Alias for scale_to_snr for backward compatibility"""
        return self.scale_to_snr(scaled)
    
    def forward(self, predictions, targets):
        """
        Compute MSE loss between predicted and target SNR values.
        
        Args:
            predictions (torch.Tensor): Model predictions (N, 1) in [0,1] range
            targets (torch.Tensor): True SNR values (N,) in [-20,30] range
            
        Returns:
            torch.Tensor: MSE loss value
        """
        # Scale targets to [0,1] range
        targets = self.scale_to_01(targets).unsqueeze(1)
        return self.mse(predictions, targets)

class DynamicLossBalancing(nn.Module):
    """
    Dynamic loss balancing for multi-task learning.
    Automatically adjusts weights based on loss ratios.
    """
    def __init__(self, num_tasks=2, alpha=0.3, eps=1e-8, device='cuda'):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha  # Increased from 0.5 to 0.3 for faster adaptation
        self.eps = eps
        self.device = device
        self.min_weight = 0.1  # Ensure no task gets completely ignored
        
        # Initialize with equal weights
        self.register_buffer('loss_ratios', torch.ones(num_tasks, device=device))
        
    def forward(self, losses):
        # Ensure all losses are on the correct device
        losses = [loss.to(self.device) for loss in losses]
        
        # Update loss ratios with exponential moving average
        current_losses = torch.tensor([loss.item() for loss in losses], device=self.device)
        self.loss_ratios = (1 - self.alpha) * self.loss_ratios + self.alpha * current_losses
        
        # Calculate weights based on loss ratios
        weights = 1.0 / (self.loss_ratios + self.eps)
        
        # Apply softmax to get normalized weights
        weights = F.softmax(weights, dim=0)
        
        # Apply minimum weight constraint
        weights = torch.clamp(weights, min=self.min_weight)
        weights = weights / weights.sum()  # Renormalize
        
        # Calculate total loss
        total_loss = sum(w * l for w, l in zip(weights, losses))
        return total_loss, weights.tolist()
    
    def get_weights(self):
        """
        Get the current task weights.
        
        Returns:
            numpy.ndarray: Normalized task weights
        """
        weights = 1.0 / (self.loss_ratios + self.eps)
        weights = weights / weights.sum()
        return weights.detach().cpu().numpy()

class KendallUncertaintyWeighting(nn.Module):
    """
    Uncertainty-based loss weighting for multi-task learning.
    Implementation based on Kendall et al. 2018 paper.
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    https://arxiv.org/abs/1705.07115
    """
    def __init__(self, num_tasks=2, device='cuda'):
        super().__init__()
        self.num_tasks = num_tasks
        self.device = device
        
        # Initialize log variances with different values to break symmetry
        # This helps prevent the weights from getting stuck at 0.5
        self.log_vars = nn.Parameter(torch.tensor([0.0, 0.5], device=device))
        
    def forward(self, losses):
        # Ensure all losses are on the correct device
        losses = [loss.to(self.device) for loss in losses]
        
        # Calculate precision (1/variance) terms
        precisions = torch.exp(-self.log_vars)
        
        # Calculate weighted losses using precision and add absolute regularization term
        weighted_losses = []
        for i, loss in enumerate(losses):
            weighted_losses.append(precisions[i] * loss + 0.5 * torch.abs(self.log_vars[i]))
        
        # Sum all weighted losses
        total_loss = sum(weighted_losses)
        
        # Calculate weights for reporting
        weights = precisions / torch.sum(precisions)
        
        return total_loss, weights.tolist()
    
    def get_weights(self):
        """
        Get the current task weights.
        
        Returns:
            numpy.ndarray: Task weights based on uncertainty
        """
        precisions = torch.exp(-self.log_vars)
        weights = precisions / torch.sum(precisions)
        return weights.detach().cpu().numpy() 