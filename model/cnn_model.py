"""
Lightweight CNN Model for Leaf Disease Classification

This module defines a simple Convolutional Neural Network trained from scratch
for classifying leaf diseases. The model is designed to:
- Run efficiently on CPU
- Accept 224x224 RGB images as input
- Output predictions for 6 classes (5 diseases + 1 healthy)
- Be easily serializable for federated learning communication

Model Architecture:
- 4 convolutional blocks (each with conv, relu, maxpool)
- Progressive feature extraction with increasing filters (32→64→128→256)
- Adaptive average pooling for flexible input handling
- Fully connected layers with dropout for regularization
- No pretrained weights - trained from scratch
"""

import torch
import torch.nn as nn


class LeafDiseaseNet(nn.Module):
    """
    Lightweight CNN for leaf disease classification.

    Args:
        num_classes (int): Number of disease classes. Default is 6 (5 diseases + healthy).
        dropout_rate (float): Dropout probability for regularization. Default is 0.5.
    """

    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(LeafDiseaseNet, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Block 1: Input -> 32 filters
        # Input: [B, 3, 224, 224]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                               padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: [B, 32, 112, 112]

        # Block 2: 32 -> 64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: [B, 64, 56, 56]

        # Block 3: 64 -> 128 filters
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: [B, 128, 28, 28]

        # Block 4: 128 -> 256 filters
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               padding=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: [B, 256, 14, 14]

        # Adaptive pooling to flatten spatial dimensions
        # Converts [B, 256, 14, 14] -> [B, 256, 1, 1]
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layers with dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # First FC layer: 256 -> 128 features
        self.fc1 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.relu_fc1 = nn.ReLU(inplace=True)

        # Second FC layer: 128 -> num_classes
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes, bias=True)

        # Initialize weights using Kaiming initialization for better training
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize convolutional and linear layer weights using Kaiming initialization.
        Biases are initialized to zero. This helps with training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for fully connected layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]
                             representing RGB images.

        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes].
                         Can be passed to CrossEntropyLoss for training
                         or softmax for inference.
        """
        # Convolutional blocks with pooling
        x = self.pool1(self.relu1(self.conv1(x)))  # [B, 32, 112, 112]
        x = self.pool2(self.relu2(self.conv2(x)))  # [B, 64, 56, 56]
        x = self.pool3(self.relu3(self.conv3(x)))  # [B, 128, 28, 28]
        x = self.pool4(self.relu4(self.conv4(x)))  # [B, 256, 14, 14]

        # Adaptive pooling to flatten spatial dimensions
        x = self.adaptive_pool(x)  # [B, 256, 1, 1]

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # [B, 256]

        # Fully connected layers with dropout (for regularization)
        x = self.dropout(x)
        x = self.relu_fc1(self.fc1(x))  # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_classes]

        return x


def get_model(num_classes=6, device='cpu'):
    """
    Convenience function to create and move model to device.

    Args:
        num_classes (int): Number of disease classes. Default is 6.
        device (str or torch.device): Device to place model on ('cpu' or 'cuda').

    Returns:
        LeafDiseaseNet: Model moved to the specified device.
    """
    model = LeafDiseaseNet(num_classes=num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    """
    Quick test to verify model architecture and parameter count.
    """
    print("=" * 60)
    print("LeafDiseaseNet Model Test")
    print("=" * 60)

    # Create model
    model = get_model(num_classes=6, device='cpu')

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / (1024**2):.2f} MB (fp32)")

    # Test forward pass with dummy input
    print(f"\nForward Pass Test:")
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    print(f"  Input Shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Output Shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")

    print("\n" + "=" * 60)
    print("Model test completed successfully!")
    print("=" * 60)
