"""
Multi-task uncertainty-weighted loss implementation based on Kendall et al. (2018).
"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Implements automatic loss weighting using homoscedastic uncertainty.
    
    The loss for each task is weighted by learned uncertainty parameters:
    L = (1/2σ²)L_task + log(σ)
    
    This allows the model to automatically balance multiple objectives
    without manual hyperparameter tuning.
    """
    
    def __init__(self, num_tasks=2):
        """
        Initialize the multi-task loss.
        
        Args:
            num_tasks (int): Number of tasks to balance
        """
        super().__init__()
        
        # Initialize log variance parameters (one per task)
        # We learn log(σ²) for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.num_tasks = num_tasks
        
    def forward(self, losses):
        """
        Compute uncertainty-weighted total loss.
        
        Args:
            losses (dict): Dictionary of individual task losses
            
        Returns:
            total_loss (torch.Tensor): Weighted total loss
            weighted_losses (dict): Individual weighted losses
            task_weights (torch.Tensor): Normalized task weights for logging
        """
        if isinstance(losses, dict):
            # Convert dict to list in consistent order
            loss_keys = list(losses.keys())
            loss_list = [losses[key] for key in loss_keys]
        else:
            loss_list = losses
            loss_keys = [f'task_{i}' for i in range(len(losses))]
        
        # Compute weighted losses
        total_loss = 0
        weighted_losses = {}
        
        for i, (loss, key) in enumerate(zip(loss_list, loss_keys)):
            # Get precision (1/σ²) from log variance
            precision = torch.exp(-self.log_vars[i])
            
            # Weighted loss: (1/2σ²)L + log(σ)
            # Note: log(σ) = 0.5 * log(σ²) = 0.5 * log_var
            weighted_loss = precision * loss + self.log_vars[i]
            
            total_loss += weighted_loss
            weighted_losses[key] = weighted_loss
        
        # Compute normalized weights for interpretability
        # These represent the relative importance of each task
        with torch.no_grad():
            # Convert log variances to actual weights (1/σ²)
            precisions = torch.exp(-self.log_vars)
            # Normalize to sum to 1
            task_weights = F.softmax(torch.log(precisions), dim=0)
        
        return total_loss, weighted_losses, task_weights
    
    def get_task_weights(self):
        """
        Get the current task weights as a dictionary.
        
        Returns:
            dict: Task weights normalized to sum to 1
        """
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars)
            weights = F.softmax(torch.log(precisions), dim=0)
            return weights.cpu().numpy()


class WeightedMultiTaskLoss(nn.Module):
    """
    Simple weighted multi-task loss with fixed weights.
    Useful for comparison with uncertainty weighting.
    """
    
    def __init__(self, task_weights=None):
        """
        Initialize with fixed task weights.
        
        Args:
            task_weights (list or dict): Fixed weights for each task
        """
        super().__init__()
        
        if task_weights is None:
            task_weights = [1.0, 1.0]  # Equal weights by default
        
        if isinstance(task_weights, dict):
            self.weights = task_weights
        else:
            self.weights = {f'task_{i}': w for i, w in enumerate(task_weights)}
    
    def forward(self, losses):
        """
        Compute weighted total loss with fixed weights.
        
        Args:
            losses (dict): Dictionary of individual task losses
            
        Returns:
            total_loss (torch.Tensor): Weighted total loss
            weighted_losses (dict): Individual weighted losses
            task_weights (torch.Tensor): Fixed task weights
        """
        total_loss = 0
        weighted_losses = {}
        
        for key, loss in losses.items():
            weight = self.weights.get(key, 1.0)
            weighted_loss = weight * loss
            total_loss += weighted_loss
            weighted_losses[key] = weighted_loss
        
        # Return weights as tensor for consistency
        weight_values = torch.tensor(list(self.weights.values()))
        normalized_weights = weight_values / weight_values.sum()
        
        return total_loss, weighted_losses, normalized_weights