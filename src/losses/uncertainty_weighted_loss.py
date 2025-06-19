# src/losses/uncertainty_weighted_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnalyticalUncertaintyWeightedLoss(nn.Module):
    """
    Analytical Uncertainty-Based Loss Weighting for Multi-Task Learning
    
    Based on the 2024 paper: "Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning"
    https://arxiv.org/abs/2408.07985
    
    This method computes analytically optimal uncertainty-based weights normalized by a 
    softmax function with tunable temperature, providing a more efficient alternative 
    to combinatorial optimization approaches.
    """
    
    def __init__(self, num_tasks=2, temperature=1.0, device='cuda'):
        super(AnalyticalUncertaintyWeightedLoss, self).__init__()
        
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.device = device
        
        # Learnable uncertainty parameters (log variance for numerical stability)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))
        
        # Store task losses for uncertainty computation
        self.task_losses = []
        
    def forward(self, task_losses):
        """
        Compute uncertainty-weighted loss
        
        Args:
            task_losses: List of individual task losses [loss_1, loss_2, ...]
            
        Returns:
            weighted_loss: Combined loss with analytical uncertainty weighting
            weights: The computed task weights for monitoring
        """
        # Convert to tensor if needed
        if isinstance(task_losses, list):
            losses = torch.stack(task_losses)
        else:
            losses = task_losses
            
        # Compute uncertainty weights using analytical solution
        # Weight = exp(-log_var) normalized by softmax with temperature
        neg_log_vars = -self.log_vars / self.temperature
        weights = F.softmax(neg_log_vars, dim=0)
        
        # Compute weighted loss with regularization term
        weighted_losses = weights * losses
        
        # Add uncertainty regularization term (proper formulation for analytical weighting)
        # The regularization should encourage learning of uncertainties, not penalize them
        regularization = 0.5 * torch.sum(torch.exp(self.log_vars))
        
        total_loss = torch.sum(weighted_losses) + regularization
        
        # Validation: Ensure loss is mathematically valid
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"Invalid loss value: {total_loss}")
        if total_loss < 0:
            print(f"Warning: Negative total loss detected: {total_loss.item():.6f}")
            print(f"  Weighted losses: {torch.sum(weighted_losses).item():.6f}")
            print(f"  Regularization: {regularization.item():.6f}")
            print(f"  Log vars: {self.log_vars.detach().cpu().numpy()}")
        
        return total_loss, weights.detach()
    
    def get_uncertainties(self):
        """Return current uncertainty values (exp of log_vars)"""
        return torch.exp(self.log_vars).detach()


class DistancePenalizedSNRLoss(nn.Module):
    """
    Distance-based loss for discrete SNR prediction that penalizes 
    predictions farther from the true SNR value.
    
    For SNR values in 2dB intervals from -20dB to 30dB (26 classes):
    SNR_values = [-20, -18, -16, ..., 28, 30]
    """
    
    def __init__(self, snr_min=-20, snr_max=30, snr_step=2, alpha=1.0, beta=0.5):
        super(DistancePenalizedSNRLoss, self).__init__()
        
        self.snr_min = snr_min
        self.snr_max = snr_max  
        self.snr_step = snr_step
        self.alpha = alpha  # Weight for cross-entropy loss
        self.beta = beta    # Weight for distance penalty
        
        # Create SNR value mapping
        self.snr_values = list(range(snr_min, snr_max + 1, snr_step))
        self.num_classes = len(self.snr_values)
        
        # Precompute distance matrix for efficiency
        self.register_buffer('distance_matrix', self._compute_distance_matrix())
        
        # Standard cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def _compute_distance_matrix(self):
        """Precompute distance matrix between all SNR classes"""
        distances = torch.zeros(self.num_classes, self.num_classes)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                snr_i = self.snr_values[i]
                snr_j = self.snr_values[j]
                distances[i, j] = abs(snr_i - snr_j) / self.snr_step
                
        return distances
    
    def forward(self, predictions, targets):
        """
        Compute distance-penalized SNR loss
        
        Args:
            predictions: Model predictions [batch_size, num_snr_classes]
            targets: True SNR class indices [batch_size]
            
        Returns:
            loss: Combined cross-entropy + distance penalty loss
        """
        batch_size = predictions.size(0)
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Distance penalty term
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Vectorized computation of distance penalty
        # Gather distance values for each target class
        target_distances = self.distance_matrix[targets]  # [batch_size, num_classes]
        
        # Compute weighted distance penalty for each sample
        distance_penalty = torch.sum(probs * target_distances, dim=1)
            
        # Combine losses
        total_loss = self.alpha * ce_loss + self.beta * distance_penalty
        
        return total_loss.mean()
    
    def snr_class_to_value(self, snr_class):
        """Convert SNR class index to actual SNR value"""
        if isinstance(snr_class, torch.Tensor):
            snr_class = snr_class.item()
        return self.snr_values[snr_class]
    
    def snr_value_to_class(self, snr_value):
        """Convert SNR value to class index"""
        # Find closest SNR value
        closest_idx = min(range(len(self.snr_values)), 
                         key=lambda i: abs(self.snr_values[i] - snr_value))
        return closest_idx