# src/losses/distance_weighted_ce.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceWeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with inverse square distance penalty for SNR classification.
    Penalizes predictions based on how far they are from the true SNR value.
    
    The penalty uses inverse square law: closer predictions get minimal penalty,
    while distant predictions get heavily penalized.
    """
    
    def __init__(self, num_classes, alpha=0.5, min_penalty=0.1):
        """
        Args:
            num_classes: Number of SNR classes (16 for 0-30 dB in 2dB steps)
            alpha: Weight for distance penalty term (0.5 = equal weight with CE loss)
            min_penalty: Minimum penalty to avoid division by zero for same-class predictions
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.min_penalty = min_penalty
        
        # Create distance matrix (assumes SNR classes are ordered 0, 2, 4, ..., 30 dB)
        # Distance in terms of class indices (0, 1, 2, ..., 15)
        indices = torch.arange(num_classes).float()
        self.register_buffer('distance_matrix', self._create_distance_matrix(indices))
        
    def _create_distance_matrix(self, indices):
        """Create matrix of distances between all class pairs."""
        # Each class represents 2dB step, so actual dB distance is 2 * index_distance
        dist_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        return dist_matrix
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output logits [batch_size, num_classes]
            targets: True SNR class indices [batch_size]
        
        Returns:
            Weighted loss combining cross-entropy and distance penalty
        """
        batch_size = predictions.size(0)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Get predicted classes (for distance penalty)
        pred_probs = F.softmax(predictions, dim=1)
        
        # Calculate expected distance based on prediction probabilities
        # This is differentiable and considers the full distribution
        target_distances = self.distance_matrix[targets]  # [batch_size, num_classes]
        
        # Calculate distance penalty using inverse square law
        # Add min_penalty to avoid division by zero
        distance_weights = 1.0 / (target_distances + self.min_penalty) ** 2
        
        # Normalize weights so they sum to 1 for each sample
        distance_weights = distance_weights / distance_weights.sum(dim=1, keepdim=True)
        
        # Calculate penalty: high probability on distant classes gets penalized
        # We want to minimize the probability mass on distant classes
        distance_penalty = -torch.sum(pred_probs * torch.log(distance_weights + 1e-8), dim=1)
        
        # Combine losses
        total_loss = ce_loss + self.alpha * distance_penalty
        
        return total_loss.mean()
    
    def get_class_weights(self, target_class):
        """
        Get the distance-based weights for a specific target class.
        Useful for visualization and debugging.
        """
        distances = self.distance_matrix[target_class]
        weights = 1.0 / (distances + self.min_penalty) ** 2
        return weights / weights.sum()