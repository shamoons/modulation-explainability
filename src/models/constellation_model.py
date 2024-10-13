# src/models/constellation_model.py
import torch
import torch.nn as nn
from torchvision import models


class ConstellationResNet(nn.Module):
    """
    A wrapper around a ResNet model from torchvision, customized for the constellation classification task.
    """

    def __init__(self, num_classes=11, input_channels=3):
        """
        Initialize the ConstellationResNet model.

        Args:
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            pretrained (bool): Whether to use a pretrained ResNet model.
        """
        super(ConstellationResNet, self).__init__()

        # Load a ResNet model from torchvision
        self.model = models.resnet18()

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

        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
