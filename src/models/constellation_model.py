# src/models/constellation_model.py
import torch.nn as nn
from torchvision import models


class ConstellationResNet(nn.Module):
    def __init__(self, num_classes=11, snr_classes=26, input_channels=3, dropout_prob=0.6, model_name="resnet18"):
        super(ConstellationResNet, self).__init__()

        # Load a ResNet model from torchvision
        if model_name == "resnet18":
            self.model = models.resnet18(weights='DEFAULT')
        elif model_name == "resnet34":
            self.model = models.resnet34(weights='DEFAULT')
        self.model_name = model_name

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
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Shared feature transformation layer
        self.shared_transform = nn.Linear(in_features, in_features // 4)

        # Single ReLU, BatchNorm, and Dropout layers
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(in_features // 4)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Separate transformation and output heads for modulation and SNR tasks
        self.modulation_head = nn.Linear(in_features // 4, num_classes)
        self.snr_head = nn.Linear(in_features // 4, snr_classes)  # Output probabilities for each SNR class

    def forward(self, x):
        features = self.model(x)

        # Shared layer processing
        features = self.relu(self.shared_transform(features))
        features = self.batch_norm(features)
        features = self.dropout(features)

        # Output heads
        modulation_output = self.modulation_head(features)
        snr_output = self.snr_head(features)  # This will output probabilities for each SNR class

        return modulation_output, snr_output
