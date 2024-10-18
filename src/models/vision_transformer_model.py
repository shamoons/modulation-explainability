# src/models/vision_transformer_model.py
import torch.nn as nn
from torchvision.models import vit_b_16


class ConstellationVisionTransformer(nn.Module):
    """
    A wrapper around a Vision Transformer (ViT) model from torchvision, customized for the constellation classification task.
    The model outputs two things:
    1) Modulation classification
    2) SNR prediction
    """

    def __init__(self, num_classes=11, snr_classes=26, input_channels=3, dropout_prob=0.5):
        """
        Initialize the ConstellationVisionTransformer model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes (26 in your case).
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.5).
        """
        super(ConstellationVisionTransformer, self).__init__()

        # Load a Vision Transformer (ViT) model from torchvision
        self.model = vit_b_16(weights='DEFAULT')
        self.model_name = "vit_b_16"

        # Modify the input embedding layer to accept the specified number of input channels
        if input_channels != 3:
            conv_proj = self.model.conv_proj
            self.model.conv_proj = nn.Conv2d(
                input_channels,
                conv_proj.out_channels,
                kernel_size=conv_proj.kernel_size,
                stride=conv_proj.stride,
                padding=conv_proj.padding,
                bias=conv_proj.bias is not None,  # Ensure bias is handled correctly as a boolean
            )

        # Extract the number of input features to the final fully connected (fc) layer
        in_features = self.model.heads.head.in_features

        # Remove the existing fully connected layer
        self.model.heads.head = nn.Identity()

        # Shared feature transformation
        self.shared_transform = nn.Linear(in_features, in_features)
        self.shared_activation = nn.ReLU()

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(in_features)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_prob)

        # Separate feature transformation layers for modulation and SNR tasks
        self.modulation_transform = nn.Linear(in_features, in_features)
        self.snr_transform = nn.Linear(in_features, in_features)

        # Output heads for modulation and SNR
        self.modulation_head = nn.Linear(in_features, num_classes)  # Modulation classification head
        self.snr_head = nn.Linear(in_features, snr_classes)  # SNR classification head

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (modulation output, snr output)
        """
        # Extract shared features using Vision Transformer
        features = self.model(x)

        # Apply batch normalization to the shared features
        features = self.batch_norm(features)

        # Apply shared transformation and activation
        features = self.shared_activation(self.shared_transform(features))

        # Apply dropout for regularization
        features = self.dropout(features)

        # Separate transformations for modulation and SNR tasks with activation
        modulation_features = self.shared_activation(self.modulation_transform(features))
        snr_features = self.shared_activation(self.snr_transform(features))

        # Output heads
        modulation_output = self.modulation_head(modulation_features)  # Predict modulation class
        snr_output = self.snr_head(snr_features)  # Predict SNR class

        return modulation_output, snr_output
