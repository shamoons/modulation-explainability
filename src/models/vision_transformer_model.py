# src/models/vision_transformer_model.py
import torch.nn as nn
from torchvision.models import vit_b_16


class ConstellationVisionTransformer(nn.Module):
    """
    A simplified wrapper around a Vision Transformer (ViT) model from torchvision, customized for constellation classification.
    The model outputs two things:
    1) Modulation classification
    2) SNR prediction
    """

    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.2, model_name="vit_b_16"):
        """
        Initialize the ConstellationVisionTransformer model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes.
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.2).
        """
        super(ConstellationVisionTransformer, self).__init__()

        # Load a Vision Transformer (ViT) model from torchvision
        self.model = vit_b_16(weights='DEFAULT')
        self.model_name = model_name

        # Modify the input embedding layer to accept the specified number of input channels
        if input_channels != 3:
            conv_proj = self.model.conv_proj
            self.model.conv_proj = nn.Conv2d(
                input_channels,
                conv_proj.out_channels,
                kernel_size=conv_proj.kernel_size,
                stride=conv_proj.stride,
                padding=conv_proj.padding,
                bias=conv_proj.bias is not None,
            )

        # Extract the number of input features to the final fully connected (fc) layer
        in_features = self.model.heads.head.in_features

        # Remove the existing fully connected layer
        self.model.heads.head = nn.Identity()

        # Shared feature transformation layer
        self.shared_transform = nn.Linear(in_features, in_features // 4)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(in_features // 4)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Output heads for modulation and SNR
        self.modulation_head = nn.Linear(in_features // 4, num_classes)
        self.snr_head = nn.Linear(in_features // 4, snr_classes)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (modulation output, snr output)
        """
        # Extract features using Vision Transformer
        features = self.model(x)

        # Apply shared transformation, activation, batch normalization, and dropout
        features = self.relu(self.shared_transform(features))
        features = self.batch_norm(features)
        features = self.dropout(features)

        # Output heads
        modulation_output = self.modulation_head(features)  # Predict modulation class
        snr_output = self.snr_head(features)  # Predict SNR class

        return modulation_output, snr_output
