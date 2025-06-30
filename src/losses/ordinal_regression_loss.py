# src/losses/ordinal_regression_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for SNR prediction.
    
    Treats SNR prediction as regression in continuous space [0, num_classes-1],
    then discretizes for evaluation. This maintains ordinal relationships while
    avoiding black hole attractors from pure classification or pure L1 distance.
    
    The model outputs a single continuous value that represents the SNR level.
    """
    
    def __init__(self, num_classes):
        """
        Args:
            num_classes: Number of ordinal classes (e.g., 16 for SNR 0-30 dB)
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_class_value = float(num_classes - 1)
        
    def forward(self, predictions, targets):
        """
        Calculate ordinal regression loss.
        
        Args:
            predictions: Model output logits [batch_size, num_classes] or [batch_size, 1]
            targets: True class indices [batch_size] with values 0 to num_classes-1
            
        Returns:
            MSE loss in continuous ordinal space
        """
        # If predictions are class logits, convert to single regression value
        if predictions.dim() == 2 and predictions.size(1) > 1:
            # Use softmax-weighted average of class indices as regression target
            class_probs = F.softmax(predictions, dim=1)
            class_indices = torch.arange(self.num_classes, device=predictions.device).float()
            pred_continuous = torch.sum(class_probs * class_indices, dim=1)
        else:
            # Single regression output - squeeze and scale
            pred_continuous = predictions.squeeze()
        
        # Ensure predictions are in valid range [0, num_classes-1]
        pred_continuous = torch.clamp(pred_continuous, 0.0, self.max_class_value)
        
        # Convert targets to float for MSE
        target_continuous = targets.float()
        
        # MSE loss in ordinal space
        loss = F.mse_loss(pred_continuous, target_continuous)
        
        return loss
    
    def predict_class(self, predictions):
        """
        Convert continuous predictions to discrete class indices.
        
        Args:
            predictions: Model output (either logits or continuous values)
            
        Returns:
            Predicted class indices
        """
        if predictions.dim() == 2 and predictions.size(1) > 1:
            # Convert from logits to continuous
            class_probs = F.softmax(predictions, dim=1)
            class_indices = torch.arange(self.num_classes, device=predictions.device).float()
            pred_continuous = torch.sum(class_probs * class_indices, dim=1)
        else:
            pred_continuous = predictions.squeeze()
            
        # Round to nearest class
        pred_continuous = torch.clamp(pred_continuous, 0.0, self.max_class_value)
        pred_classes = torch.round(pred_continuous).long()
        
        return pred_classes


if __name__ == "__main__":
    # Test ordinal regression loss
    num_snr_classes = 16  # 0-30 dB in 2dB steps
    loss_fn = OrdinalRegressionLoss(num_classes=num_snr_classes)
    
    # Test with multi-class logits
    batch_size = 5
    logits = torch.randn(batch_size, num_snr_classes)
    targets = torch.tensor([0, 3, 8, 12, 15])  # Various SNR classes
    
    loss = loss_fn(logits, targets)
    print(f"Ordinal Regression Loss: {loss.item():.4f}")
    
    # Get predicted classes
    pred_classes = loss_fn.predict_class(logits)
    print(f"True classes: {targets.tolist()}")
    print(f"Predicted classes: {pred_classes.tolist()}")
    
    # Test with single regression output
    single_output = torch.randn(batch_size, 1) * num_snr_classes
    loss_single = loss_fn(single_output, targets)
    print(f"\nSingle output loss: {loss_single.item():.4f}")