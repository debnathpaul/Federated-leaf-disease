"""
Global model utilities for federated learning.

This module provides utilities for managing the global model state,
including loading, saving, and evaluating on test data.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LeafDiseaseNet


class GlobalModelManager:
    """
    Manages global model throughout federated learning training.

    Handles:
    - Model initialization
    - Weight updates (after global FedAvg)
    - Checkpoint saving/loading
    - Evaluation on test set
    """

    def __init__(self, num_classes=6, device='cpu'):
        """
        Args:
            num_classes (int): Number of disease classes
            device (str): 'cpu' or 'cuda'
        """
        self.num_classes = num_classes
        self.device = device
        self.model = LeafDiseaseNet(num_classes=num_classes, dropout_rate=0.5)
        self.model = self.model.to(device)
        self.round = 0

    def get_model(self):
        """Get current model."""
        return self.model

    def get_weights(self):
        """Get current model weights as dictionary."""
        return self.model.state_dict()

    def set_weights(self, weights_dict):
        """
        Update model with new weights.

        Args:
            weights_dict (dict): State dictionary with weights
        """
        self.model.load_state_dict(weights_dict)
        self.model = self.model.to(self.device)

    def evaluate(self, test_loader, num_classes=6):
        """
        Evaluate model on test dataset.

        Args:
            test_loader (DataLoader): Test data loader
            num_classes (int): Number of classes

        Returns:
            dict: Evaluation metrics {accuracy, loss, per_class_acc}
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        # Per-class accuracy
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)

                total_loss += loss.item()
                _, predictions = torch.max(logits.data, 1)

                total += batch_y.size(0)
                correct += (predictions == batch_y).sum().item()

                # Per-class metrics
                for i in range(num_classes):
                    mask = batch_y == i
                    class_total[i] += mask.sum().item()
                    class_correct[i] += (predictions[mask] == batch_y[mask]).sum().item()

        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        per_class_acc = []
        for i in range(num_classes):
            acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            per_class_acc.append(acc)

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'per_class_accuracy': per_class_acc,
            'total_samples': total
        }

    def save_checkpoint(self, filepath, metadata=None):
        """
        Save model checkpoint.

        Args:
            filepath (str): Path to save checkpoint
            metadata (dict): Additional metadata to save
        """
        checkpoint = {
            'round': self.round,
            'model_state': self.model.state_dict(),
            'num_classes': self.num_classes,
            'metadata': metadata or {}
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.

        Args:
            filepath (str): Path to checkpoint file

        Returns:
            dict: Metadata from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.round = checkpoint.get('round', 0)
        metadata = checkpoint.get('metadata', {})

        self.model = self.model.to(self.device)

        print(f"Checkpoint loaded: {filepath}")
        print(f"  Round: {self.round}")

        return metadata

    def update_round(self, round_num):
        """Update current round number."""
        self.round = round_num

    def get_model_size_mb(self):
        """Get model size in MB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        size_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
        return size_mb

    def summary(self):
        """Print model summary."""
        print(f"\n{'='*70}")
        print(f"Global Model Summary")
        print(f"{'='*70}")
        print(f"Classes: {self.num_classes}")
        print(f"Device: {self.device}")
        print(f"Round: {self.round}")
        print(f"Size: {self.get_model_size_mb():.2f} MB")
        print(f"\nModel Architecture:")
        print(self.model)
        print(f"{'='*70}\n")


def create_global_model(num_classes=6, device='cpu'):
    """
    Factory function to create global model manager.

    Args:
        num_classes (int): Number of classes
        device (str): Device to use

    Returns:
        GlobalModelManager: Initialized model manager
    """
    return GlobalModelManager(num_classes=num_classes, device=device)


if __name__ == "__main__":
    # Example usage
    print("Creating global model...")
    model_manager = create_global_model(num_classes=6, device='cpu')
    model_manager.summary()

    print(f"Model size: {model_manager.get_model_size_mb():.2f} MB")
