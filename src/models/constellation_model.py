# src/models/constellation_model.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from .task_specific_extractor import TaskSpecificFeatureExtractor


class ConstellationResNet(nn.Module):
    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.3, model_name="resnet18", snr_layer_config="standard"):
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

        # Task-specific feature extractors that create different representations
        self.task_specific_extractor = TaskSpecificFeatureExtractor(
            input_dim=in_features,
            task_dim=in_features // 4,
            dropout_prob=dropout_prob
        )

        # Separate transformation and output heads for modulation and SNR tasks
        self.modulation_head = nn.Linear(in_features // 4, num_classes)
        
        # Configure SNR head based on snr_layer_config
        snr_input_dim = in_features // 4
        if snr_layer_config == "standard":
            # Direct linear layer (no bottleneck)
            self.snr_head = nn.Linear(snr_input_dim, snr_classes)
        elif snr_layer_config == "bottleneck_64":
            # 64-dimensional bottleneck
            self.snr_head = nn.Sequential(
                nn.Linear(snr_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(64, snr_classes)
            )
        elif snr_layer_config == "bottleneck_128":
            # 128-dimensional bottleneck
            self.snr_head = nn.Sequential(
                nn.Linear(snr_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(128, snr_classes)
            )
        elif snr_layer_config == "dual_layer":
            # Two-layer architecture with 256->64 compression
            self.snr_head = nn.Sequential(
                nn.Linear(snr_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(64, snr_classes)
            )
        else:
            raise ValueError(f"Unknown snr_layer_config: {snr_layer_config}")

    def forward(self, x):
        # Extract features using ResNet backbone
        shared_features = self.model(x)

        # Task-specific feature extraction and fusion
        mod_features, snr_features = self.task_specific_extractor(shared_features)

        # Output heads
        modulation_output = self.modulation_head(mod_features)
        snr_output = self.snr_head(snr_features)

        return modulation_output, snr_output
