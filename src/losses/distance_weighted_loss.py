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
        
        # Normalize distances to [0, 1] range first
        max_distance = self.num_classes - 1
        normalized_distances = distances / max_distance
        
        # Apply non-linear curve for better adjacent penalty
        # Option 1: Square root curve - less aggressive than linear, more than quadratic
        # normalized_penalty = torch.sqrt(normalized_distances)
        
        # Option 2: Logarithmic curve - strong penalty for adjacent, plateaus for distant
        # epsilon = 1e-8  # Avoid log(0)
        # normalized_penalty = -torch.log(1 - normalized_distances + epsilon) / -torch.log(epsilon)
        
        # Option 3: Sigmoid-based curve - smooth transition with configurable steepness
        steepness = 5.0  # Higher = steeper curve around midpoint
        normalized_penalty = 2 / (1 + torch.exp(-steepness * normalized_distances)) - 1
        
        # Apply alpha scaling
        distance_penalty = self.alpha * normalized_penalty
        
        # Additional scaling to keep total loss reasonable
        # Target: distance penalty should add at most 0.1-0.3 to CE loss
        # Since sigmoid maxes at ~1.0, and alpha=0.5, max penalty is ~0.5
        # Scale down by factor of 2 to get max ~0.25
        distance_penalty = distance_penalty / 2.0
        
        # Average the distance penalty across the batch
        distance_penalty = torch.mean(distance_penalty)
        
        # Combined loss
        total_loss = ce_loss + distance_penalty
        
        return total_loss