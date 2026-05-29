"""
Evaluation metrics for leaf disease classification.

Computes:
- Accuracy (overall and per-class)
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-score (macro and weighted)
- Confusion matrix

Important for imbalanced/non-IID data where per-class metrics matter.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)


def compute_metrics(y_true, y_pred, num_classes=6):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true (list or array): True class labels
        y_pred (list or array): Predicted class labels
        num_classes (int): Number of classes

    Returns:
        dict: Dictionary with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Macro metrics (unweighted average across classes)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted metrics (weighted by support in each class)
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist()
    }

    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a readable format.

    Args:
        metrics (dict): Metrics dictionary from compute_metrics()
        class_names (list): Optional class names for per-class metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluation Metrics")
    print(f"{'='*70}")

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    print(f"\nMacro-Averaged (unweighted across classes):")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall: {metrics['macro_recall']:.4f}")
    print(f"  F1-Score: {metrics['macro_f1']:.4f}")

    print(f"\nWeighted-Averaged (by class support):")
    print(f"  Precision: {metrics['weighted_precision']:.4f}")
    print(f"  Recall: {metrics['weighted_recall']:.4f}")
    print(f"  F1-Score: {metrics['weighted_f1']:.4f}")

    # Per-class metrics
    num_classes = len(metrics['per_class_f1'])
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*51}")

    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        precision = metrics['per_class_precision'][i]
        recall = metrics['per_class_recall'][i]
        f1 = metrics['per_class_f1'][i]
        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

    print(f"\n{'='*70}\n")


def print_confusion_matrix(cm, class_names=None):
    """
    Print confusion matrix.

    Args:
        cm (array-like): Confusion matrix
        class_names (list): Optional class names
    """
    cm = np.array(cm)
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]

    print(f"\nConfusion Matrix:")
    print(f"{'Predicted →':<12}", end='')
    for name in class_names:
        print(f"{name[:8]:>10}", end='')
    print()

    print(f"{'Actual ↓':<12}", end='')
    for _ in class_names:
        print(f"{'-'*10}", end='')
    print()

    for i, true_label in enumerate(class_names):
        print(f"{true_label:<12}", end='')
        for j in range(num_classes):
            print(f"{cm[i, j]:>10}", end='')
        print()

    print()


def balanced_accuracy(y_true, y_pred, num_classes=6):
    """
    Compute balanced accuracy (average recall across classes).

    Good for imbalanced datasets.

    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        num_classes (int): Number of classes

    Returns:
        float: Balanced accuracy
    """
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    balanced_acc = np.mean(recall_per_class)
    return float(balanced_acc)


def class_distribution(labels, class_names=None):
    """
    Print class distribution in dataset.

    Args:
        labels (array): Class labels
        class_names (list): Optional class names
    """
    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print(f"\nClass Distribution:")
    print(f"{'Class':<15} {'Samples':<10} {'Percentage':<10}")
    print(f"{'-'*35}")

    total = len(labels)
    for i in range(num_classes):
        count = np.sum(labels == i)
        percentage = 100 * count / total
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{class_name:<15} {count:<10} {percentage:>6.1f}%")

    print()


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0])

    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    print_metrics(metrics, class_names=['Healthy', 'Disease1', 'Disease2'])

    print_confusion_matrix(metrics['confusion_matrix'],
                          class_names=['Healthy', 'Disease1', 'Disease2'])

    balanced_acc = balanced_accuracy(y_true, y_pred, num_classes=3)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
