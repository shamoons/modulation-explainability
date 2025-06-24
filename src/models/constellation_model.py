# src/models/constellation_model.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models


class TaskSpecificFeatureExtractor(nn.Module):
    """
    Task-specific feature extraction that creates different representations for each task
    before applying task heads. This prevents task competition at the feature level.
    
    Uses different transformations and attention mechanisms to create task-specific
    feature spaces, then applies residual connections and gating to preserve information.
    """
    
    def __init__(self, input_dim, task_dim, dropout_prob=0.3):
        super().__init__()
        
        # Different activation functions to create different feature distributions
        self.mod_branch = nn.Sequential(
            nn.Linear(input_dim, task_dim),
            nn.GELU(),  # Different activation for modulation task
            nn.BatchNorm1d(task_dim),
            nn.Dropout(dropout_prob)
        )
        
        self.snr_branch = nn.Sequential(
            nn.Linear(input_dim, task_dim),
            nn.ReLU(),  # Different activation for SNR task
            nn.BatchNorm1d(task_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Task-specific attention mechanisms
        self.mod_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.ReLU(),
            nn.Linear(input_dim // 8, input_dim),
            nn.Sigmoid()
        )
        
        self.snr_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.Tanh(),  # Different activation
            nn.Linear(input_dim // 8, input_dim),
            nn.Sigmoid()
        )
        
        # Residual projections to preserve information
        self.mod_residual = nn.Linear(input_dim, task_dim) if input_dim != task_dim else nn.Identity()
        self.snr_residual = nn.Linear(input_dim, task_dim) if input_dim != task_dim else nn.Identity()
        
    def forward(self, shared_features):
        """
        Extract task-specific features using different attention patterns and transformations.
        
        Args:
            shared_features: Features from the backbone model
            
        Returns:
            tuple: (modulation_features, snr_features)
        """
        # Apply task-specific attention to shared features
        mod_attended = shared_features * self.mod_attention(shared_features)
        snr_attended = shared_features * self.snr_attention(shared_features)
        
        # Transform through task-specific branches
        mod_features = self.mod_branch(mod_attended)
        snr_features = self.snr_branch(snr_attended)
        
        # Add residual connections to preserve information
        mod_residual = self.mod_residual(shared_features)
        snr_residual = self.snr_residual(shared_features)
        
        # Weighted combination instead of simple addition
        mod_features = 0.7 * mod_features + 0.3 * mod_residual
        snr_features = 0.7 * snr_features + 0.3 * snr_residual
        
        return mod_features, snr_features


class ConstellationResNet(nn.Module):
    def __init__(self, num_classes=20, snr_classes=26, input_channels=1, dropout_prob=0.3, model_name="resnet18"):
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
        # SNR head: 26 classes for discrete SNR values (-20 to 30 dB in 2dB steps)
        self.snr_head = nn.Linear(in_features // 4, snr_classes)

    def forward(self, x):
        # Extract features using ResNet backbone
        shared_features = self.model(x)

        # Task-specific feature extraction and fusion
        mod_features, snr_features = self.task_specific_extractor(shared_features)

        # Output heads
        modulation_output = self.modulation_head(mod_features)
        snr_output = self.snr_head(snr_features)

        return modulation_output, snr_output
