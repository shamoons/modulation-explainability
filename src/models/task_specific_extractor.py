# src/models/task_specific_extractor.py
"""
Task-specific feature extraction module for multi-task learning.

This module provides the TaskSpecificFeatureExtractor class that creates
different representations for each task before applying task heads. This
prevents task competition at the feature level by using different
transformations and attention mechanisms.

Key features:
- Different activation functions for modulation vs SNR tasks
- Task-specific attention mechanisms to focus on relevant features
- Residual connections with weighted combination to preserve information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificFeatureExtractor(nn.Module):
    """
    Task-specific feature extraction that creates different representations for each task
    before applying task heads. This prevents task competition at the feature level.
    
    Uses different transformations and attention mechanisms to create task-specific
    feature spaces, then applies residual connections and gating to preserve information.
    
    Args:
        input_dim (int): Dimension of input features from backbone model
        task_dim (int): Dimension of output task-specific features
        dropout_prob (float): Dropout probability for regularization
    
    Returns:
        tuple: (modulation_features, snr_features) - Task-specific feature representations
    """
    
    def __init__(self, input_dim, task_dim, dropout_prob=0.3):
        super().__init__()
        
        # Different activation functions to create different feature distributions
        self.mod_branch = nn.Sequential(
            nn.Linear(input_dim, task_dim),
            nn.GELU(),  # Different activation for modulation task
            nn.BatchNorm1d(task_dim),
            nn.Dropout(dropout_prob)
        )
        
        self.snr_branch = nn.Sequential(
            nn.Linear(input_dim, task_dim),
            nn.ReLU(),  # Different activation for SNR task
            nn.BatchNorm1d(task_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Task-specific attention mechanisms
        self.mod_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.ReLU(),
            nn.Linear(input_dim // 8, input_dim),
            nn.Sigmoid()
        )
        
        self.snr_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.Tanh(),  # Different activation
            nn.Linear(input_dim // 8, input_dim),
            nn.Sigmoid()
        )
        
        # Residual projections to preserve information
        self.mod_residual = nn.Linear(input_dim, task_dim) if input_dim != task_dim else nn.Identity()
        self.snr_residual = nn.Linear(input_dim, task_dim) if input_dim != task_dim else nn.Identity()
        
    def forward(self, shared_features):
        """
        Extract task-specific features using different attention patterns and transformations.
        
        Args:
            shared_features (torch.Tensor): Features from the backbone model
            
        Returns:
            tuple: (modulation_features, snr_features)
                - modulation_features: Features optimized for modulation classification
                - snr_features: Features optimized for SNR prediction/classification
        """
        # Apply task-specific attention to shared features
        mod_attended = shared_features * self.mod_attention(shared_features)
        snr_attended = shared_features * self.snr_attention(shared_features)
        
        # Transform through task-specific branches
        mod_features = self.mod_branch(mod_attended)
        snr_features = self.snr_branch(snr_attended)
        
        # Add residual connections to preserve information
        mod_residual = self.mod_residual(shared_features)
        snr_residual = self.snr_residual(shared_features)
        
        # Weighted combination instead of simple addition
        # 70% task-specific, 30% shared to balance specialization and preservation
        mod_features = 0.7 * mod_features + 0.3 * mod_residual
        snr_features = 0.7 * snr_features + 0.3 * snr_residual
        
        return mod_features, snr_features