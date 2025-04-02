# src/models/constellation_model.py
import torch.nn as nn
from torchvision import models


class ConstellationResNet(nn.Module):
    def __init__(self, num_classes, snr_classes, input_channels=1, model_name="resnet18"):
        super(ConstellationResNet, self).__init__()
        self.model_name = model_name
        
        # Load pre-trained ResNet model
        if model_name == "resnet18":
            self.resnet = models.resnet18(pretrained=False)
        elif model_name == "resnet50":
            self.resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Modify first layer to accept single channel input
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add dropout
        self.dropout = nn.Dropout(0.5)
        
        # Add modulation classification head
        self.modulation_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Add SNR classification head
        self.snr_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, snr_classes)
        )
        
    def forward(self, x):
        # Get features from ResNet
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        
        # Apply dropout to features
        features = self.dropout(features)
        
        # Get predictions from both heads
        modulation_output = self.modulation_head(features)
        snr_output = self.snr_head(features)
        
        return modulation_output, snr_output
