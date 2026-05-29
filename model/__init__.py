"""Model package for federated learning leaf disease detection."""

from .cnn_model import LeafDiseaseNet, get_model

__all__ = ['LeafDiseaseNet', 'get_model']
