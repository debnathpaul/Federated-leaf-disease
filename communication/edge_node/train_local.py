"""
Local training script for edge node (farm).

Each farm trains the CNN model on its local non-IID dataset in isolation.
After training, weights are sent to the regional aggregator.
This implements the "E" in FedAvg (local epochs of training).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Fix Windows terminal encoding issues
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.cnn_model import LeafDiseaseNet
from communication.edge_node.dataset import load_dataset, get_data_loaders
from communication.edge_node.send_weights import send_weights_to_aggregator
from utils.metrics import compute_metrics


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): CNN model to train
        train_loader (DataLoader): Training data loader
        optimizer: Adam or SGD optimizer
        criterion: Loss function (CrossEntropyLoss)
        device: CPU or CUDA device

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        # Move data to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation set.

    Args:
        model (nn.Module): CNN model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: CPU or CUDA device

    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item()
            _, predictions = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predictions == batch_y).sum().item()

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train_local(farm_id, dataset_root, num_epochs=5, batch_size=32,
                learning_rate=0.001, num_classes=6, device='cpu'):
    """
    Full local training pipeline for a farm node.

    Args:
        farm_id (int): Farm identifier (0 to num_farms-1)
        dataset_root (str): Path to PlantVillage dataset root
        num_epochs (int): Number of local training epochs per round
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        num_classes (int): Number of disease classes
        device (str): 'cpu' or 'cuda'

    Returns:
        tuple: (trained_model, metrics_dict)
    """
    print(f"\n{'='*70}")
    print(f"Farm {farm_id}: Starting Local Training")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")

    # 1. Load dataset and create data loaders
    print(f"\n[1/3] Loading dataset...")
    all_image_paths, all_labels, class_names = load_dataset(dataset_root, num_classes)
    print(f"  Total samples loaded: {len(all_image_paths)}")
    print(f"  Classes: {class_names}")

    train_loader, val_loader = get_data_loaders(
        all_image_paths, all_labels, farm_id, num_farms=6,
        batch_size=batch_size, num_workers=0
    )
    num_train_samples = len(train_loader.dataset)
    print(f"  Farm {farm_id} - Train samples: {num_train_samples}, "
          f"Val samples: {len(val_loader.dataset)}")

    # 2. Create model and training components
    print(f"\n[2/3] Initializing model and optimizer...")
    model = LeafDiseaseNet(num_classes=num_classes, dropout_rate=0.5)
    model = model.to(device)
    print(f"  Model moved to device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training loop
    print(f"\n[3/3] Training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"  Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

    # 4. Final validation with detailed metrics
    print(f"\nFinal Evaluation:")
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device)

    # Compute detailed metrics
    model.eval()
    all_preds = []
    all_labels_eval = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            _, preds = torch.max(logits.data, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_eval.extend(batch_y.numpy())

    metrics = compute_metrics(all_labels_eval, all_preds, num_classes)
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    # Return trained model and metrics
    metrics_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_val_accuracy': final_val_acc,
        'final_val_loss': final_val_loss,
        'detailed_metrics': metrics,
        'num_train_samples': num_train_samples
    }

    print(f"\n{'='*70}")
    print(f"Farm {farm_id}: Training Complete")
    print(f"{'='*70}\n")

    return model, metrics_dict


def get_model_weights(model):
    """
    Extract model weights as a dictionary (for serialization/sending).

    Args:
        model (nn.Module): PyTorch model

    Returns:
        dict: Model state dictionary with weights
    """
    return model.state_dict()


def set_model_weights(model, weights_dict, device='cpu'):
    """
    Load weights into model from dictionary.

    Args:
        model (nn.Module): PyTorch model to update
        weights_dict (dict): State dictionary with weights
        device: Device to load weights to

    Returns:
        nn.Module: Updated model
    """
    model.load_state_dict(weights_dict)
    model = model.to(device)
    return model


def get_aggregator_url_for_farm(farm_id):
    """
    Map farm ID to regional aggregator URL.

    Farm to Region mapping:
        - Farm 0, 1 -> Region A (port 5001)
        - Farm 2, 3 -> Region B (port 5002)
        - Farm 4, 5 -> Region C (port 5003)

    Args:
        farm_id (int): Farm identifier

    Returns:
        tuple: (region_name, aggregator_url)
    """
    farm_to_region = {
        0: ('A', 'http://localhost:5001'),
        1: ('A', 'http://localhost:5001'),
        2: ('B', 'http://localhost:5002'),
        3: ('B', 'http://localhost:5002'),
        4: ('C', 'http://localhost:5003'),
        5: ('C', 'http://localhost:5003'),
    }

    if farm_id not in farm_to_region:
        raise ValueError(f"Unknown farm_id: {farm_id}")

    region, url = farm_to_region[farm_id]
    return region, url


if __name__ == "__main__":
    """
    Local training script for farm node (can be run standalone or as subprocess).

    Run with: python communication/edge_node/train_local.py [options]
    Example: python communication/edge_node/train_local.py --farm_id 0 --epochs 2
    """
    import argparse

    parser = argparse.ArgumentParser(description="Local training for farm node")
    parser.add_argument('--farm_id', type=int, default=0, help='Farm identifier')
    parser.add_argument('--round', type=int, default=1, help='FL round number')
    parser.add_argument('--dataset_root', type=str, default='data/plantvillage/', help='Dataset root path')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--aggregator_url', type=str, default='http://localhost:5001',
                       help='Aggregator URL (e.g., http://localhost:5001)')

    args = parser.parse_args()

    # Training configuration
    farm_id = args.farm_id
    round_number = args.round
    dataset_root = args.dataset_root
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    device = args.device
    aggregator_url_arg = args.aggregator_url

    print(f"\nTraining Configuration:")
    print(f"  Farm ID: {farm_id}")
    print(f"  Round Number: {round_number}")
    print(f"  Dataset: {dataset_root}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Device: {device}")

    # Start training
    print(f"\nStarting training...\n")
    model, metrics = train_local(
        farm_id=farm_id,
        dataset_root=dataset_root,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_classes=6,
        device=device
    )

    # Extract number of training samples from metrics
    num_train_samples = metrics.get('num_train_samples', 1000)

    # Print summary
    print(f"\n" + "="*70)
    print(f"Training Complete - Farm {farm_id}")
    print(f"="*70)

    print(f"\nLoss and Accuracy by Epoch:")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Val Accuracy':<15}")
    print(f"{'-'*53}")

    for epoch in range(num_epochs):
        train_loss = metrics['train_losses'][epoch]
        val_loss = metrics['val_losses'][epoch]
        val_acc = metrics['val_accuracies'][epoch]
        print(f"{epoch+1:<8} {train_loss:<15.4f} {val_loss:<15.4f} {val_acc:<15.4f}")

    print(f"\nFinal Results:")
    print(f"  Final Validation Accuracy: {metrics['final_val_accuracy']:.4f}")
    print(f"  Final Validation Loss: {metrics['final_val_loss']:.4f}")

    print(f"\nDetailed Metrics:")
    detailed = metrics['detailed_metrics']
    print(f"  Macro Precision: {detailed['macro_precision']:.4f}")
    print(f"  Macro Recall: {detailed['macro_recall']:.4f}")
    print(f"  Macro F1-Score: {detailed['macro_f1']:.4f}")

    # Send trained weights to regional aggregator
    print(f"\n" + "="*70)
    print(f"Sending weights to aggregator...")
    print(f"Aggregator URL: {aggregator_url_arg}")
    print(f"="*70)

    try:
        weights = get_model_weights(model)

        response = send_weights_to_aggregator(
            farm_id=farm_id,
            weights_dict=weights,
            aggregator_url=aggregator_url_arg,
            round_number=round_number,
            num_samples=num_train_samples,
            retries=3,
            timeout=60
        )
        print(f"\nWeights sent successfully!")
        print(f"Response: {response}")
    except RuntimeError as e:
        print(f"\n[ERROR] Failed to send weights: {e}")
        print(f"Aggregator may not be running or unreachable.")

    print(f"\n" + "="*70)
    print(f"[OK] Training test passed!")
    print(f"="*70 + "\n")
