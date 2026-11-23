"""
SarcasTone — Multimodal Fusion Model
======================================
THE core model. Combines text embeddings + audio prosodic features
through feature-level concatenation and a shared classifier.

Architecture (matches PPT Slide 5-6):
  TextEmbedding(768) ──┐
                       ├── Concatenate → Normalize → Dense(256) → ReLU → Dropout
  AudioFeatures(26) ───┘                            → Dense(128) → ReLU → Dropout
                                                     → Dense(1) → Sigmoid
"""

import torch
import torch.nn as nn
from src.config import TEXT_EMBEDDING_DIM, AUDIO_FEATURE_DIM, FUSION_HIDDEN_DIM, DROPOUT_RATE


class FusionModel(nn.Module):
    """Multimodal feature-level fusion classifier for sarcasm detection."""

    def __init__(self, text_dim: int = TEXT_EMBEDDING_DIM,
                 audio_dim: int = AUDIO_FEATURE_DIM,
                 hidden_dim: int = FUSION_HIDDEN_DIM,
                 dropout: float = DROPOUT_RATE):
        super().__init__()

        combined_dim = text_dim + audio_dim  # 768 + 26 = 794

        # Normalization layer for the combined features
        self.layer_norm = nn.LayerNorm(combined_dim)

        # Fusion classification network
        self.network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),        # 794 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),     # 256 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),              # 128 → 1
        )

    def forward(self, text_embedding: torch.Tensor,
                audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embedding: (batch, 768) DistilBERT [CLS] embeddings.
            audio_features: (batch, 26) prosodic feature vectors.
        Returns:
            (batch, 1) raw logits.
        """
        # Feature-level fusion: concatenate
        combined = torch.cat([text_embedding, audio_features], dim=1)

        # Normalize the combined representation
        combined = self.layer_norm(combined)

        # Classify
        return self.network(combined)
