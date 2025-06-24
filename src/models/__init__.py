# src/models/__init__.py
"""
Models package for constellation diagram-based automatic modulation classification.

This package contains model architectures for multi-task learning with task-specific
feature extraction, supporting both modulation classification and SNR prediction.
"""

from .constellation_model import ConstellationResNet
from .vision_transformer_model import ConstellationVisionTransformer  
from .swin_transformer_model import ConstellationSwinTransformer
from .task_specific_extractor import TaskSpecificFeatureExtractor

__all__ = [
    'ConstellationResNet',
    'ConstellationVisionTransformer', 
    'ConstellationSwinTransformer',
    'TaskSpecificFeatureExtractor'
]