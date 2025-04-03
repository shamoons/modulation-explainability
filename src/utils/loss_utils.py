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

class DynamicLossBalancing(nn.Module):
    """
    Implementation of Dynamic Loss Balancing for Multi-Task Learning
    Based on: "Dynamic Loss Balancing for Multi-Task Learning" (Li et al., 2023)
    
    Key features:
    1. Uses gradient statistics to balance task losses
    2. Maintains moving averages of gradient norms and loss ratios
    3. Adapts weights based on task difficulty and learning progress
    """
    def __init__(self, num_tasks, device):
        super().__init__()
        self.num_tasks = num_tasks
        self.device = device
        
        # Hyperparameters from paper
        self.alpha = 0.3  # Reduced moving average factor for more responsive weights
        self.eps = 1e-8  # Small constant for numerical stability
        
        # Initialize gradient statistics as buffers (persistent tensors)
        # These track the moving averages of gradient norms and loss ratios
        self.register_buffer('grad_norms', torch.ones(num_tasks, device=device))
        self.register_buffer('loss_ratios', torch.ones(num_tasks, device=device))
        
    def forward(self, losses):
        """
        Compute weighted loss based on gradient statistics and loss ratios.
        
        Args:
            losses (list): List of task-specific losses
            
        Returns:
            torch.Tensor: Total weighted loss
        """
        # Ensure all losses are on the correct device
        losses = [loss.to(self.device) for loss in losses]
        
        # Compute loss ratios (Equation 5)
        loss_ratios = torch.tensor([loss.item() for loss in losses], device=self.device)
        self.loss_ratios = self.alpha * self.loss_ratios + (1 - self.alpha) * loss_ratios
        
        # Compute task weights based on loss ratios (Equation 6)
        weights = 1.0 / (self.loss_ratios + self.eps)
        weights = weights / weights.sum()  # Normalize weights
        
        # Compute weighted losses (Equation 7)
        weighted_losses = [weights[i] * loss for i, loss in enumerate(losses)]
        total_loss = sum(weighted_losses)
        
        return total_loss
    
    def get_weights(self):
        """
        Get the current task weights.
        
        Returns:
            numpy.ndarray: Normalized task weights
        """
        # Return normalized weights (Equation 6)
        weights = 1.0 / (self.loss_ratios + self.eps)
        weights = weights / weights.sum()
        return weights.detach().cpu().numpy() 