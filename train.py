"""
SarcasTone — Training Script
==============================
Trains all 3 models (text-only, audio-only, fusion) and saves weights.
Generates evaluation metrics and comparative results.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

from src.config import (
    MODEL_DIR, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
)
from src.text_encoder import get_text_encoder
from src.audio_prosody import extract_prosody
from src.data_loader import load_dataset, generate_samples, create_dataloaders
from src.models.text_baseline import TextBaseline
from src.models.audio_baseline import AudioBaseline
from src.models.fusion_model import FusionModel
from src.evaluate import evaluate_model, print_comparison


def prepare_features(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract text embeddings and audio features for all samples.

    Returns:
        text_embeddings: (N, 768)
        audio_features: (N, 26)
        labels: (N,)
    """
    text_encoder = get_text_encoder()

    texts = [s["text"] for s in samples]
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)

    # Batch encode text
    print("[Train] Encoding text with DistilBERT...")
    text_embeddings = text_encoder.encode_batch(texts)
    print(f"[Train] Text embeddings shape: {text_embeddings.shape}")

    # Extract audio features
    print("[Train] Extracting audio prosodic features...")
    audio_feats = []
    for s in samples:
        feats = extract_prosody(audio_path=s["audio_path"])
        audio_feats.append(feats)
    audio_features = torch.tensor(np.array(audio_feats), dtype=torch.float32)
    print(f"[Train] Audio features shape: {audio_features.shape}")

    return text_embeddings, audio_features, labels


def train_model(model: nn.Module, train_loader, val_loader,
                model_name: str, input_mode: str = "fusion",
                epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE) -> nn.Module:
    """
    Train a single model.

    Args:
        model: The neural network model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        model_name: Name for saving weights.
        input_mode: 'text', 'audio', or 'fusion'.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    print(f"\n{'='*50}")
    print(f"Training: {model_name} ({input_mode} mode)")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if input_mode == "text":
                inputs = batch["text_embedding"].to(device)
                logits = model(inputs)
            elif input_mode == "audio":
                inputs = batch["audio_features"].to(device)
                logits = model(inputs)
            else:  # fusion
                text_emb = batch["text_embedding"].to(device)
                audio_feat = batch["audio_features"].to(device)
                logits = model(text_emb, audio_feat)

            labels = batch["label"].to(device).unsqueeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if input_mode == "text":
                    inputs = batch["text_embedding"].to(device)
                    logits = model(inputs)
                elif input_mode == "audio":
                    inputs = batch["audio_features"].to(device)
                    logits = model(inputs)
                else:
                    text_emb = batch["text_embedding"].to(device)
                    audio_feat = batch["audio_features"].to(device)
                    logits = model(text_emb, audio_feat)

                labels = batch["label"].to(device).unsqueeze(1)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1) * 100
        val_acc = val_correct / max(val_total, 1) * 100
        avg_val_loss = val_loss / max(len(val_loader), 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Acc: {train_acc:5.1f}% | "
                  f"Val Acc: {val_acc:5.1f}% | "
                  f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved: {save_path}")

    return model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  SarcasTone — Training Pipeline")
    print("=" * 60)

    # Step 1: Generate samples
    print("\n[1/5] Generating training samples...")
    generate_samples(force=True)

    # Step 2: Load data
    print("\n[2/5] Loading dataset...")
    samples = load_dataset()
    print(f"  Loaded {len(samples)} samples")

    # Step 3: Extract features
    print("\n[3/5] Extracting features...")
    text_emb, audio_feat, labels = prepare_features(samples)

    # Step 4: Create data loaders
    print("\n[4/5] Creating data splits...")
    train_loader, val_loader, test_loader = create_dataloaders(
        text_emb, audio_feat, labels, batch_size=BATCH_SIZE
    )

    # Step 5: Train all 3 models
    print("\n[5/5] Training models...")

    # Text-only baseline
    text_model = TextBaseline()
    text_model = train_model(text_model, train_loader, val_loader,
                             "text_baseline", input_mode="text")

    # Audio-only baseline
    audio_model = AudioBaseline()
    audio_model = train_model(audio_model, train_loader, val_loader,
                              "audio_baseline", input_mode="audio")

    # Fusion model
    fusion_model = FusionModel()
    fusion_model = train_model(fusion_model, train_loader, val_loader,
                               "fusion_model", input_mode="fusion")

    # Step 6: Evaluate all models on test set
    print("\n" + "=" * 60)
    print("  EVALUATION ON TEST SET")
    print("=" * 60)

    text_results = evaluate_model(text_model, test_loader, "Text-Only Baseline", "text")
    audio_results = evaluate_model(audio_model, test_loader, "Audio-Only Baseline", "audio")
    fusion_results = evaluate_model(fusion_model, test_loader, "Fusion Model (SarcasTone)", "fusion")

    # Print comparison table
    print_comparison(text_results, audio_results, fusion_results)

    print("\n✅ Training complete! Models saved to saved_models/")
    print("   Run 'python app.py' to launch the demo.")


if __name__ == "__main__":
    main()
