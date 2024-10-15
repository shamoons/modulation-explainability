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
        # Get predicted class by finding the index of the max logits (no softmax here)
        snr_pred_class = torch.argmax(snr_pred, dim=1)

        # Compute the distance between predicted and true classes
        distance_penalty = (snr_pred_class.float() - snr_true.float()).abs()

        # Apply a scaling factor to penalize further distances
        scaled_penalty = distance_penalty

        # Standard cross-entropy loss (logits passed directly)
        ce_loss = nn.CrossEntropyLoss()(snr_pred, snr_true)

        # Final loss is cross-entropy loss with an additional penalty term
        loss = ce_loss + torch.mean(scaled_penalty)

        return loss


if __name__ == "__main__":
    # Simulate random predicted logits for 5 samples and 10 SNR classes (before softmax)
    snr_pred = torch.randn(5, 10)  # 5 samples, 10 classes (logits)

    # Simulate true SNR class labels for these samples
    snr_true = torch.tensor([0, 2, 5, 7, 9])  # True SNR class indices

    # Instantiate the custom loss function
    loss_fn = DistancePenaltyCategoricalSNRLoss()

    # Calculate the loss
    loss = loss_fn(snr_pred, snr_true)

    # Print the computed loss value
    print(f"Computed loss: {loss.item()}")
