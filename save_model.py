"""
Train and save the leaf disease model locally.
Run this once to save the model, then use predict.py for predictions.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.cnn_model import LeafDiseaseNet
from communication.edge_node.dataset import load_dataset, get_data_loaders

DATASET_ROOT = 'data/plantvillage/'
MODEL_SAVE_PATH = 'model/leaf_disease_model.pth'
NUM_CLASSES = 6
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001

CLASS_NAMES = [
    'Tomato_healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Potato___Early_blight',
    'Potato___Late_blight'
]

def train_and_save():
    print("="*60)
    print("Leaf Disease Model Training & Saving")
    print("="*60)

    # Load dataset
    print("\n[1] Loading dataset...")
    try:
        all_image_paths, all_labels, class_names = load_dataset(DATASET_ROOT, NUM_CLASSES)
        train_loader, val_loader = get_data_loaders(
            all_image_paths, all_labels,
            farm_id=0, num_farms=1,
            batch_size=BATCH_SIZE, num_workers=0
        )
        print(f"    Dataset loaded: {len(all_image_paths)} images")
    except Exception as e:
        print(f"    [ERROR] Could not load dataset: {e}")
        return

    # Initialize model
    print("\n[2] Initializing model...")
    model = LeafDiseaseNet(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"    Model ready")

    # Train
    print(f"\n[3] Training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"    Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        acc = 100. * correct / total
        print(f"  >> Epoch {epoch+1} done | Accuracy: {acc:.2f}% | Loss: {total_loss/len(train_loader):.4f}")

    # Save model
    print(f"\n[4] Saving model to '{MODEL_SAVE_PATH}'...")
    os.makedirs('model', exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES
    }, MODEL_SAVE_PATH)

    print(f"    [OK] Model saved!")
    print("\n" + "="*60)
    print("Done! Now run: python predict.py <image_path>")
    print("="*60)

if __name__ == "__main__":
    train_and_save()