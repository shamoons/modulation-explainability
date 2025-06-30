# src/losses/distance_weighted_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceWeightedSNRLoss(nn.Module):
    """
    Distance-weighted cross-entropy loss for SNR classification.
    
    Combines standard cross-entropy with a distance penalty that penalizes
    predictions far from the true SNR class. This maintains ordinal relationships
    (22 dB is closer to 24 dB than to 30 dB) while preserving discrete classification.
    
    CORRECTED IMPLEMENTATION: Higher distance = higher penalty (not backwards!)
    """
    
    def __init__(self, num_classes, alpha=0.5):
        """
        Initialize distance-weighted SNR loss.
        
        Args:
            num_classes (int): Number of SNR classes
            alpha (float): Weight for distance penalty (default: 0.5)
        """
        super(DistanceWeightedSNRLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        """
        Compute distance-weighted loss.
        
        Args:
            predictions (torch.Tensor): Model predictions (logits) [batch_size, num_classes]
            targets (torch.Tensor): True class indices [batch_size]
            
        Returns:
            torch.Tensor: Combined loss (cross-entropy + distance penalty)
        """
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Get predicted classes (for distance calculation)
        predicted_classes = torch.argmax(predictions, dim=1)
        
        # Calculate distance penalty
        # Distance = |predicted_class - true_class|
        distances = torch.abs(predicted_classes.float() - targets.float())
        
        # Distance penalty = alpha * distance^2
        # Quadratic penalty: adjacent errors (distance=1) get penalty=alpha,
        # distant errors (distance=4) get penalty=16*alpha
        distance_penalty = self.alpha * (distances ** 2)
        
        # Average the distance penalty across the batch
        distance_penalty = torch.mean(distance_penalty)
        
        # Combined loss
        total_loss = ce_loss + distance_penalty
        
        return total_loss