"""
Curriculum learning package for SNR classification.

Contains modules for curriculum stage management, dataset filtering,
and configuration for progressive learning.
"""

from .curriculum_manager import CurriculumManager
from .curriculum_dataset import CurriculumAwareDataset
from .curriculum_config import CURRICULUM_STAGES, DEFAULT_CURRICULUM_PATIENCE

__all__ = [
    'CurriculumManager',
    'CurriculumAwareDataset',
    'CURRICULUM_STAGES',
    'DEFAULT_CURRICULUM_PATIENCE'
] 