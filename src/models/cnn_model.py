# models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCBlock(nn.Module):
    """
    A depthwise separable convolution block.
    Depthwise convolution followed by pointwise convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSCBlock, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(LightweightCNN, self).__init__()

        # First convolution block (change in_channels to 2 for I/Q channels)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Depthwise separable convolution blocks
        self.dsc_block1 = DSCBlock(32, 64)
        self.dsc_block2 = DSCBlock(64, 128)
        self.dsc_block3 = DSCBlock(128, 256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Pass through depthwise separable convolution blocks
        x = self.dsc_block1(x)
        x = self.dsc_block2(x)
        x = self.dsc_block3(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Example of creating the model and printing the architecture
    model = LightweightCNN(num_classes=11)
    print(model)

    # Example of input tensor (Batch size, Channels, Height, Width)
    # Example input (Batch size: 8, Image size: 64x64)
    sample_input = torch.randn(8, 1, 64, 64)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
