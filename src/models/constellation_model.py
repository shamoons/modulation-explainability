# src/models/constellation_model.py
import torch.nn as nn
from torchvision import models


class ConstellationResNet(nn.Module):
    """
    A wrapper around a ResNet model from torchvision, customized for the constellation classification task.
    The model outputs two things:
    1) Modulation classification
    2) SNR prediction
    """

    def __init__(self, num_classes=11, snr_classes=26, input_channels=3, dropout_prob=0.5):
        """
        Initialize the ConstellationResNet model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes (26 in your case).
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            dropout_prob (float): Probability of dropout (defaults to 0.5).
        """
        super(ConstellationResNet, self).__init__()

        # Load a ResNet model from torchvision
        self.model = models.resnet18(weights='DEFAULT')
        self.model_name = "resnet18"

        # Modify the first convolutional layer to accept the specified number of input channels
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                input_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias,
            )

        # Remove the fully connected layer of ResNet
        in_features = self.model.fc.in_features  # Number of input features to the final FC layer
        self.model.fc = nn.Identity()  # Replace the fully connected layer with an identity operation

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
        # Extract shared features using ResNet
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
