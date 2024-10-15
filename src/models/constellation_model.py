# src/models/constellation_model.py
import torch
import torch.nn as nn
from torchvision import models


class ConstellationResNet(nn.Module):
    """
    A wrapper around a ResNet model from torchvision, customized for the constellation classification task.
    The model outputs two things: 
    1) Modulation classification
    2) SNR prediction
    """

    def __init__(self, num_classes=11, snr_classes=26, input_channels=3):
        """
        Initialize the ConstellationResNet model with two output heads.

        Args:
            num_classes (int): Number of output classes for modulation.
            snr_classes (int): Number of possible SNR classes (26 in your case).
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
        """
        super(ConstellationResNet, self).__init__()

        # Load a ResNet model from torchvision
        self.model = models.resnet34(weights='DEFAULT')

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

        # Remove the fully connected layer of ResNet50
        in_features = self.model.fc.in_features  # Number of input features to the final FC layer
        self.model.fc = nn.Identity()  # Replace the fully connected layer with an identity operation

        # Add two new fully connected layers:
        # One for modulation classification and one for SNR prediction
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
        features = self.model(x)  # Extract features using ResNet
        modulation_output = self.modulation_head(features)  # Predict modulation class
        snr_output = self.snr_head(features)  # Predict SNR class
        return modulation_output, snr_output
