"""
SarcasTone — Audio-Only Baseline Model
========================================
Classifier that uses ONLY audio prosodic features (no text).
Serves as a baseline to compare against the fusion model.
Architecture: AudioFeatures(26) → Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dense(1) → Sigmoid
"""

import torch
import torch.nn as nn
from src.config import AUDIO_FEATURE_DIM, DROPOUT_RATE


class AudioBaseline(nn.Module):
    def __init__(self, input_dim: int = AUDIO_FEATURE_DIM,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, 26) prosodic feature vectors.
        Returns:
            (batch_size, 1) raw logits.
        """
        return self.network(audio_features)
