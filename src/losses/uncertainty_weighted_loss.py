# src/losses/uncertainty_weighted_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnalyticalUncertaintyWeightedLoss(nn.Module):
    """
    Homoscedastic Uncertainty-Based Loss Weighting for Multi-Task Learning
    
    Based on the highly-cited 2018 paper: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    https://arxiv.org/abs/1705.07115
    @inproceedings{kendall2018multi,
      title={Multi-task learning using uncertainty to weigh losses for scene geometry and semantics},
      author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={7482--7491},
      year={2018}
    }
    
    This method learns homoscedastic uncertainty (task-dependent uncertainty) to automatically
    balance losses in multi-task learning. The approach weighs each task's loss by the inverse
    of its learned uncertainty, preventing tasks from dominating based on their relative scales.
    
    Key advantages:
    - Well-established method with extensive validation (1000+ citations)
    - Prevents task competition through independent weighting
    - Principled uncertainty modeling with proper regularization
    """
    
    def __init__(self, num_tasks=2, temperature=1.5, device='cuda', min_weight=0.05):
        super(AnalyticalUncertaintyWeightedLoss, self).__init__()
        
        self.num_tasks = num_tasks
        self.device = device
        self.min_weight = min_weight
        # Note: temperature parameter kept for backward compatibility but not used in Kendall method
        
        # Learnable homoscedastic uncertainty parameters (log variance for numerical stability)
        # Initialize to zeros following Kendall et al. (corresponds to sigma=1 initially)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))
        
        # Store task losses for uncertainty computation
        self.task_losses = []
        
    def forward(self, task_losses):
        """
        Compute uncertainty-weighted loss using Kendall et al. formulation
        
        Args:
            task_losses: List of individual task losses [loss_1, loss_2, ...]
            
        Returns:
            weighted_loss: Combined loss with homoscedastic uncertainty weighting
            weights: The computed task weights for monitoring (derived from uncertainties)
        """
        # Convert to tensor if needed
        if isinstance(task_losses, list):
            losses = torch.stack(task_losses)
        else:
            losses = task_losses
            
        # Clip log variances to prevent numerical instability
        clipped_log_vars = torch.clamp(self.log_vars, min=-2.0, max=2.0)
        
        # Kendall et al. formulation: L = (1/(2*sigma^2)) * task_loss + 0.5 * log(sigma^2)
        # where sigma^2 = exp(log_var)
        sigmas_squared = torch.exp(clipped_log_vars)
        
        # Compute individual task weights: 1/(2*sigma^2)
        task_weights = 1.0 / (2.0 * sigmas_squared)
        
        # Apply minimum weight constraint for stability
        # Convert to relative weights for monitoring while preserving Kendall formulation
        task_weights_clamped = torch.clamp(task_weights, min=self.min_weight)
        
        # Compute weighted losses for each task
        weighted_losses = task_weights_clamped * losses
        
        # Kendall regularization term: 0.5 * sum(log(sigma^2)) = 0.5 * sum(log_vars)
        regularization = 0.5 * torch.sum(clipped_log_vars)
        
        # Total loss following Kendall et al.
        total_loss = torch.sum(weighted_losses) + regularization
        
        # Compute normalized weights for monitoring (approximation for display)
        normalized_weights = task_weights_clamped / torch.sum(task_weights_clamped)
        
        # Validation: Ensure loss is mathematically valid
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"Invalid loss value: {total_loss}")
        if total_loss < 0:
            print(f"Warning: Negative total loss detected: {total_loss.item():.6f}")
            print(f"  Weighted losses: {torch.sum(weighted_losses).item():.6f}")
            print(f"  Regularization: {regularization.item():.6f}")
            print(f"  Log vars: {self.log_vars.detach().cpu().numpy()}")
            print(f"  Sigmas squared: {sigmas_squared.detach().cpu().numpy()}")
        
        return total_loss, normalized_weights.detach()
    
    def get_uncertainties(self):
        """Return current uncertainty values (exp of clipped log_vars)"""
        clipped_log_vars = torch.clamp(self.log_vars, min=-2.0, max=2.0)
        return torch.exp(clipped_log_vars).detach()


class DistancePenalizedSNRLoss(nn.Module):
    """
    Distance-based loss for discrete SNR prediction that penalizes 
    predictions farther from the true SNR value.
    
    For SNR values in 2dB intervals from -20dB to 30dB (26 classes):
    SNR_values = [-20, -18, -16, ..., 28, 30]
    """
    
    def __init__(self, snr_min=-20, snr_max=30, snr_step=2, alpha=1.0, beta=0.5, snr_values=None):
        super(DistancePenalizedSNRLoss, self).__init__()
        
        self.snr_min = snr_min
        self.snr_max = snr_max  
        self.snr_step = snr_step
        self.alpha = alpha  # Weight for cross-entropy loss
        self.beta = beta    # Weight for distance penalty
        
        # Create SNR value mapping - use provided values or generate from range
        if snr_values is not None:
            self.snr_values = sorted(snr_values)  # Use actual SNR values from dataset
        else:
            self.snr_values = list(range(snr_min, snr_max + 1, snr_step))  # Fallback to range
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
                # Normalize distance by step size, but handle arbitrary SNR values
                if len(self.snr_values) == 1:
                    step_size = 1.0  # Avoid division by zero when only one SNR value
                else:
                    step_size = min(abs(self.snr_values[k+1] - self.snr_values[k]) for k in range(len(self.snr_values)-1))
                distances[i, j] = abs(snr_i - snr_j) / step_size
                
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