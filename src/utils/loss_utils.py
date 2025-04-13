import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSNRLoss(nn.Module):
    def __init__(self, snr_values, device):
        """
        Initialize the weighted SNR loss for classification.
        
        Args:
            snr_values (list): List of SNR values in order (e.g., [-20, 0, 30])
            device (torch.device): Device to place tensors on
        """
        super().__init__()
        self.device = device
        self.snr_values = torch.tensor(snr_values, dtype=torch.float32, device=device)
        self.num_classes = len(snr_values)
        
    def forward(self, predictions, targets):
        """
        Compute weighted cross-entropy loss for SNR classification.
        
        Args:
            predictions (torch.Tensor): Model predictions (N, num_classes)
            targets (torch.Tensor): True SNR class indices (N,)
            
        Returns:
            torch.Tensor: Weighted cross-entropy loss value
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Get predicted class indices
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Calculate distance between predicted and true classes
        pred_values = self.snr_values[pred_classes]
        true_values = self.snr_values[targets]
        
        # Calculate normalized distance weights
        value_range = self.snr_values.max() - self.snr_values.min()
        distance_weights = 1.0 + torch.abs(pred_values - true_values) / value_range
        
        # Apply weights to cross-entropy loss
        weighted_loss = distance_weights * ce_loss
        
        return weighted_loss.mean()

    def scale_to_snr(self, predictions):
        """
        Convert class predictions to SNR values for metrics.
        
        Args:
            predictions (torch.Tensor): Predictions from model [batch_size, num_classes]
            
        Returns:
            torch.Tensor: SNR values [batch_size, 1]
        """
        # Get predicted class indices
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Convert to actual SNR values
        snr_values = self.snr_values[pred_classes]
        
        return snr_values.unsqueeze(1)

    def get_snr_metrics(self, predictions, targets):
        """
        Calculate both classification accuracy and MAE for monitoring.
        
        Args:
            predictions (torch.Tensor): Predictions from model [batch_size, num_classes]
            targets (torch.Tensor): True SNR class indices [batch_size]
            
        Returns:
            tuple: (accuracy, mae)
            - accuracy: Classification accuracy (percentage)
            - mae: Mean absolute error between predicted and true SNR values
        """
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1)
        accuracy = (pred_classes == targets).float().mean().item() * 100
        
        # Calculate MAE using actual SNR values
        pred_values = self.snr_values[pred_classes]
        true_values = self.snr_values[targets]
        mae = torch.abs(pred_values - true_values).mean().item()
        
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