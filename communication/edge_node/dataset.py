"""
Dataset loader for PlantVillage leaf disease images.

This module handles loading PlantVillage dataset and creating non-IID (non-independent
and identically distributed) data splits across farm nodes to simulate real-world
federated learning scenarios where different farms grow different crop varieties.

Non-IID Distribution:
- Each farm gets a non-uniform distribution of disease classes
- Some farms specialize in certain crops/diseases
- This mimics real agricultural diversity across regions
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class PlantVillageDataset(Dataset):
    """
    Custom Dataset for PlantVillage leaf disease images.

    Attributes:
        root_dir (str): Path to PlantVillage dataset directory
        image_paths (list): List of image file paths
        labels (list): List of corresponding class labels
        transform (callable): Image transformation pipeline
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of class labels (0-5)
            transform (callable): Torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def get_disease_classes(root_dir=None):
    """
    Get list of exactly 6 disease classes for this project.

    Classes:
    0. Tomato_healthy
    1. Tomato_Bacterial_spot
    2. Tomato_Early_blight
    3. Tomato_Late_blight
    4. Potato___Early_blight
    5. Potato___Late_blight

    Args:
        root_dir (str): Path to PlantVillage dataset (unused, for compatibility)

    Returns:
        list: Ordered list of 6 disease class names
    """
    classes = [
        'Tomato_healthy',
        'Tomato_Bacterial_spot',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Potato___Early_blight',
        'Potato___Late_blight'
    ]
    return classes


def load_dataset(root_dir, num_classes=6):
    """
    Load all images and labels from PlantVillage dataset.

    Loads exactly 6 classes:
    0. Tomato_healthy
    1. Tomato_Bacterial_spot
    2. Tomato_Early_blight
    3. Tomato_Late_blight
    4. Potato___Early_blight
    5. Potato___Late_blight

    Args:
        root_dir (str): Path to PlantVillage dataset root (e.g., 'data/plantvillage/')
        num_classes (int): Must be 6 (default is 6)

    Returns:
        tuple: (all_image_paths, all_labels, class_names) where labels are 0-indexed
    """
    classes = get_disease_classes(root_dir)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    all_image_paths = []
    all_labels = []

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(img_path)
                all_labels.append(class_to_idx[class_name])

    return all_image_paths, all_labels, classes


def create_non_iid_split(all_image_paths, all_labels, farm_id, num_farms=6, concentration=0.8):
    """
    Create a non-IID (non-uniform) data split for a specific farm.

    This simulates real-world scenario where different farms grow different crops.
    Each farm gets a concentration of certain disease classes.

    Args:
        all_image_paths (list): All image paths
        all_labels (list): All labels
        farm_id (int): Farm identifier (0 to num_farms-1)
        num_farms (int): Total number of farms
        concentration (float): How concentrated each farm's data is (0.8 = 80% from main classes)

    Returns:
        tuple: (farm_image_paths, farm_labels) for this specific farm
    """
    num_classes = len(set(all_labels))

    # Organize images by class
    class_to_images = {i: [] for i in range(num_classes)}
    for img_path, label in zip(all_image_paths, all_labels):
        class_to_images[label].append(img_path)

    # Assign main classes to each farm (overlapping but different focus)
    main_classes = [(farm_id + i) % num_classes for i in range(num_classes)]

    farm_image_paths = []
    farm_labels = []

    # Add concentration% from main classes
    for class_id in main_classes:
        images_in_class = class_to_images[class_id]
        start_idx = int(farm_id * len(images_in_class) / num_farms)
        end_idx = start_idx + int(concentration * len(images_in_class) / num_farms)
        end_idx = min(end_idx, len(images_in_class))

        selected_images = images_in_class[start_idx:end_idx]
        farm_image_paths.extend(selected_images)
        farm_labels.extend([int(class_id)] * len(selected_images))

    # Add remaining (1-concentration)% from non-main classes
    non_main_images = []
    for class_id in range(num_classes):
        if class_id not in main_classes:
            non_main_images.extend(class_to_images[class_id])

    num_remaining = int((1 - concentration) * len(farm_image_paths) / concentration)
    if num_remaining > 0 and non_main_images:
        num_to_select = min(num_remaining, len(non_main_images))
        selected_indices = np.random.choice(len(non_main_images), num_to_select, replace=False).tolist()
        for idx in selected_indices:
            img_path = non_main_images[idx]
            farm_image_paths.append(img_path)
            # Find the label for this image
            for label, images in class_to_images.items():
                if img_path in images:
                    farm_labels.append(int(label))
                    break

    return farm_image_paths, farm_labels


def get_data_loaders(all_image_paths, all_labels, farm_id, num_farms=6,
                     batch_size=32, num_workers=2):
    """
    Create data loaders for a farm with non-IID split.

    Args:
        all_image_paths (list): All image paths from dataset
        all_labels (list): All labels
        farm_id (int): This farm's ID
        num_farms (int): Total number of farms
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get non-IID split for this farm
    farm_image_paths, farm_labels = create_non_iid_split(
        all_image_paths, all_labels, farm_id, num_farms
    )

    # 80-20 train-val split
    num_train = int(0.8 * len(farm_image_paths))
    indices = np.random.permutation(len(farm_image_paths))
    train_indices = indices[:num_train].tolist()  # Convert to Python list
    val_indices = indices[num_train:].tolist()    # Convert to Python list

    train_paths = [farm_image_paths[i] for i in train_indices]
    train_labels = [farm_labels[i] for i in train_indices]
    val_paths = [farm_image_paths[i] for i in val_indices]
    val_labels = [farm_labels[i] for i in val_indices]

    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = PlantVillageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantVillageDataset(val_paths, val_labels, transform=val_transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader


if __name__ == "__main__":
    """
    Test dataset loading and print statistics.
    Run with: python edge_node/dataset.py
    """
    print("\n" + "="*70)
    print("PlantVillage Dataset Test")
    print("="*70)

    # Configuration
    dataset_root = 'data/plantvillage/'
    num_classes = 6

    print(f"\nDataset Configuration:")
    print(f"  Root: {dataset_root}")
    print(f"  Number of classes: {num_classes}")

    # Load dataset
    print(f"\n[1/3] Loading dataset...")
    try:
        all_image_paths, all_labels, class_names = load_dataset(dataset_root, num_classes)
        print(f"  [OK] Dataset loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Error loading dataset: {e}")
        exit(1)

    # Print class information
    print(f"\n[2/3] Class Information:")
    print(f"  {'-'*68}")
    print(f"  {'Index':<8} {'Class Name':<40} {'Count':<12}")
    print(f"  {'-'*68}")

    class_counts = {}
    for i, class_name in enumerate(class_names):
        count = sum(1 for label in all_labels if label == i)
        class_counts[class_name] = count
        print(f"  {i:<8} {class_name:<40} {count:<12}")

    print(f"  {'-'*68}")
    print(f"  Total images found: {len(all_image_paths)}")

    # Get data loaders for Farm 0
    print(f"\n[3/3] Creating data loaders for Farm 0...")
    try:
        train_loader, val_loader = get_data_loaders(
            all_image_paths, all_labels, farm_id=0, num_farms=6,
            batch_size=32, num_workers=0
        )
        print(f"  [OK] Data loaders created successfully")
    except Exception as e:
        print(f"  [ERROR] Error creating data loaders: {e}")
        exit(1)

    print(f"\nDataset Split:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Total for Farm 0: {len(train_loader.dataset) + len(val_loader.dataset)}")

    # Test loading a batch
    print(f"\nTesting batch loading...")
    try:
        images, labels = next(iter(train_loader))
        print(f"  [OK] Batch loaded successfully")
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
    except Exception as e:
        print(f"  [ERROR] Error loading batch: {e}")
        exit(1)

    print(f"\n" + "="*70)
    print(f"[OK] All tests passed!")
    print(f"="*70 + "\n")
