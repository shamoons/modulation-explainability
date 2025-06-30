# src/losses/distance_snr_loss.py
import torch
import torch.nn as nn


class PureDistanceSNRLoss(nn.Module):
    """
    Pure L1 distance loss for ordinal SNR classification.
    
    Directly minimizes the absolute difference between predicted and true SNR classes.
    This treats SNR prediction as an ordinal regression problem rather than 
    categorical classification, eliminating black hole attractors.
    
    No cross-entropy loss, no alpha parameter - just pure distance optimization.
    """
    
    def __init__(self):
        super(PureDistanceSNRLoss, self).__init__()

    def forward(self, snr_pred, snr_true):
        """
        Calculate pure L1 distance loss between predicted and true SNR classes.

        Args:
            snr_pred (Tensor): Predicted logits for SNR classes [batch_size, num_classes]
            snr_true (Tensor): True SNR labels as class indices [batch_size]

        Returns:
            Tensor: Pure L1 distance loss (mean absolute error in class space)
        """
        # Get predicted class by finding the index of the max logits
        snr_pred_class = torch.argmax(snr_pred, dim=1)

        # Compute L1 distance between predicted and true classes
        distance_loss = torch.abs(snr_pred_class.float() - snr_true.float())

        # Return mean L1 distance
        return torch.mean(distance_loss)


if __name__ == "__main__":
    # Test pure distance loss
    snr_pred = torch.randn(5, 16)  # 5 samples, 16 SNR classes (0-30 dB in 2dB steps)
    snr_true = torch.tensor([0, 2, 8, 12, 15])  # True SNR class indices

    # Instantiate the pure distance loss
    loss_fn = PureDistanceSNRLoss()

    # Calculate the loss
    loss = loss_fn(snr_pred, snr_true)

    # Print the computed loss value
    print(f"Pure L1 Distance Loss: {loss.item()}")
    
    # Show predicted vs true for demonstration
    pred_classes = torch.argmax(snr_pred, dim=1)
    distances = torch.abs(pred_classes.float() - snr_true.float())
    print(f"Predicted classes: {pred_classes.tolist()}")
    print(f"True classes: {snr_true.tolist()}")
    print(f"L1 distances: {distances.tolist()}")
