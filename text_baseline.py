"""
SarcasTone — Text-Only Baseline Model
=======================================
Classifier that uses ONLY text embeddings (no audio).
Serves as a baseline to compare against the fusion model.
Architecture: TextEmbedding(768) → Dense(256) → ReLU → Dropout → Dense(1) → Sigmoid
"""

import torch
import torch.nn as nn
from src.config import TEXT_EMBEDDING_DIM, FUSION_HIDDEN_DIM, DROPOUT_RATE


class TextBaseline(nn.Module):
    def __init__(self, input_dim: int = TEXT_EMBEDDING_DIM,
                 hidden_dim: int = FUSION_HIDDEN_DIM,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embedding: (batch_size, 768) DistilBERT embeddings.
        Returns:
            (batch_size, 1) raw logits (apply sigmoid for probability).
        """
        return self.network(text_embedding)
