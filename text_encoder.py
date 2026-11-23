"""
SarcasTone — Text Encoder Module
=================================
Uses DistilBERT to convert text into a 768-dimensional semantic embedding.
Handles tokenization and [CLS] token extraction for sentence-level representation.
"""

import torch
from transformers import DistilBertTokenizer, DistilBertModel
from src.config import TEXT_MODEL_NAME, TEXT_EMBEDDING_DIM


class TextEncoder:
    """Encodes text sentences into fixed-size semantic embeddings using DistilBERT."""

    def __init__(self, model_name: str = TEXT_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TextEncoder] Loading {model_name} on {self.device}...")
        try:
            # Try loading cached files first (avoids network delays/offline errors)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = DistilBertModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        except Exception:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[TextEncoder] Ready. Output dim = {TEXT_EMBEDDING_DIM}")

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single text string into a 768-dim embedding vector.

        Args:
            text: Input sentence string.

        Returns:
            1D tensor of shape (768,) — the [CLS] token embedding.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # Use the [CLS] token (first token) as the sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)
        return cls_embedding.squeeze(0).cpu()  # shape: (768,)

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a batch of text strings.

        Args:
            texts: List of sentence strings.

        Returns:
            2D tensor of shape (batch_size, 768).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu()


# Singleton instance (lazy-loaded)
_encoder_instance = None


def get_text_encoder() -> TextEncoder:
    """Get or create the singleton TextEncoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = TextEncoder()
    return _encoder_instance
