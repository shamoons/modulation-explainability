# src/losses/distance_snr_loss.py
import torch
import torch.nn as nn


class DistancePenaltyCategoricalSNRLoss(nn.Module):
    def __init__(self):
        super(DistancePenaltyCategoricalSNRLoss, self).__init__()

    def forward(self, snr_pred, snr_true):
        """
        Calculate a penalty-based categorical loss for SNR where predictions further from the
        true SNR class are penalized more.

        Args:
            snr_pred (Tensor): Predicted logits for SNR classes (before softmax).
            snr_true (Tensor): True SNR labels (as class indices).

        Returns:
            Tensor: Loss value with distance-based penalties.
        """
        # Convert logits to probabilities
        snr_prob = torch.softmax(snr_pred, dim=1)

        # Get predicted class by finding the index of the max probability
        snr_pred_class = torch.argmax(snr_prob, dim=1)

        # Compute the distance between predicted and true classes
        distance_penalty = (snr_pred_class.float() - snr_true.float()).abs()

        # Apply a scaling factor to make further distances more penalized
        scaled_penalty = distance_penalty ** 2  # Squared distance

        # Standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(snr_pred, snr_true)

        # Final loss is cross-entropy loss with an additional penalty term
        loss = ce_loss + torch.mean(scaled_penalty)

        return loss
