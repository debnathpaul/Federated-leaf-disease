"""
Visualization utilities for federated learning experiments.

Creates plots for:
- Training curves (loss, accuracy over rounds/epochs)
- Per-class metrics
- Confusion matrices
- Class distribution
- Communication costs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_training_curves(train_losses, val_losses, val_accuracies,
                        title="Training Progress", save_path=None):
    """
    Plot training curves (loss and accuracy over epochs).

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        val_accuracies (list): Validation accuracies per epoch
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, val_accuracies, 'g-s', label='Validation Accuracy', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_fl_rounds(round_accuracies, round_losses=None, title="Federated Learning Progress",
                   save_path=None):
    """
    Plot metrics across federated learning rounds.

    Args:
        round_accuracies (list): Accuracy per FL round
        round_losses (list): Loss per FL round (optional)
        title (str): Plot title
        save_path (str): Path to save figure
    """
    rounds = range(1, len(round_accuracies) + 1)

    if round_losses is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Accuracy
        axes[0].plot(rounds, round_accuracies, 'g-o', linewidth=2, markersize=6)
        axes[0].set_xlabel('FL Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Global Accuracy over FL Rounds')
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(rounds, round_losses, 'r-s', linewidth=2, markersize=6)
        axes[1].set_xlabel('FL Round')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Global Loss over FL Rounds')
        axes[1].grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, round_accuracies, 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel('FL Round')
        ax.set_ylabel('Accuracy')
        ax.set_title('Global Accuracy over FL Rounds')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm (array-like): Confusion matrix (num_classes x num_classes)
        class_names (list): Optional class names
        title (str): Plot title
        save_path (str): Path to save figure
    """
    cm = np.array(cm)
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Normalize for visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'},
                ax=ax)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_per_class_metrics(precisions, recalls, f1_scores, class_names=None,
                           title="Per-Class Metrics", save_path=None):
    """
    Plot per-class metrics as bar chart.

    Args:
        precisions (list): Precision per class
        recalls (list): Recall per class
        f1_scores (list): F1-score per class
        class_names (list): Optional class names
        title (str): Plot title
        save_path (str): Path to save figure
    """
    num_classes = len(f1_scores)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    x = np.arange(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_class_distribution(labels, class_names=None, title="Class Distribution",
                           save_path=None):
    """
    Plot class distribution as bar chart.

    Args:
        labels (array): Class labels
        class_names (list): Optional class names
        title (str): Plot title
        save_path (str): Path to save figure
    """
    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    counts = [np.sum(labels == i) for i in range(num_classes)]
    percentages = [100 * c / len(labels) for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(class_names, counts, color='steelblue', alpha=0.7)

    # Add count labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Number of Samples')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


def plot_fedavg_convergence(farm_accuracies, regional_accuracies=None,
                           global_accuracy=None, title="FedAvg Convergence",
                           save_path=None):
    """
    Plot convergence of federated averaging across different levels.

    Args:
        farm_accuracies (dict): {farm_id: [accuracies per round]}
        regional_accuracies (dict): {region: [accuracies per round]} (optional)
        global_accuracy (list): Global accuracies per round (optional)
        title (str): Plot title
        save_path (str): Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot farm accuracies
    if farm_accuracies:
        for farm_id, accuracies in farm_accuracies.items():
            rounds = range(1, len(accuracies) + 1)
            ax.plot(rounds, accuracies, '--', alpha=0.5, label=f'Farm {farm_id}')

    # Plot regional averages
    if regional_accuracies:
        for region, accuracies in regional_accuracies.items():
            rounds = range(1, len(accuracies) + 1)
            ax.plot(rounds, accuracies, '-', linewidth=2, label=f'Region {region}')

    # Plot global accuracy
    if global_accuracy:
        rounds = range(1, len(global_accuracy) + 1)
        ax.plot(rounds, global_accuracy, '-o', linewidth=3, markersize=8,
               color='red', label='Global')

    ax.set_xlabel('FL Round')
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Creating example plots...")

    # Sample data
    train_losses = [2.0, 1.5, 1.2, 1.0, 0.9, 0.8]
    val_losses = [2.1, 1.6, 1.3, 1.1, 1.0, 0.9]
    val_accuracies = [0.3, 0.45, 0.55, 0.65, 0.72, 0.78]

    plot_training_curves(train_losses, val_losses, val_accuracies,
                        save_path='./plots/training_curves.png')

    # Confusion matrix
    cm = np.array([[80, 10, 5, 0, 5, 0],
                  [5, 85, 5, 0, 5, 0],
                  [5, 5, 80, 0, 5, 5],
                  [0, 0, 0, 90, 5, 5],
                  [5, 5, 5, 5, 75, 5],
                  [0, 0, 5, 5, 5, 85]])

    plot_confusion_matrix(cm, class_names=['Healthy', 'D1', 'D2', 'D3', 'D4', 'D5'],
                         save_path='./plots/confusion_matrix.png')
