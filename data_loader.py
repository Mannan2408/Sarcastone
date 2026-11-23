"""
SarcasTone — Data Loader & Sample Generator
=============================================
Generates synthetic training samples with matching text + audio
for training and testing the sarcasm detection models.

Also provides a DataLoader wrapper for PyTorch training.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf

from src.config import DATA_DIR, SAMPLE_RATE, AUDIO_FEATURE_DIM, TRAIN_SPLIT, VAL_SPLIT, BATCH_SIZE


# --- Sample utterances with sarcasm labels ---
SAMPLE_UTTERANCES = [
    # SARCASTIC examples (label = 1)
    {"text": "Oh wow, that was really amazing", "label": 1, "tone": "flat"},
    {"text": "Great job, you're a genius", "label": 1, "tone": "flat"},
    {"text": "Sure, because that always works out perfectly", "label": 1, "tone": "exaggerated"},
    {"text": "Oh fantastic, just what I needed today", "label": 1, "tone": "flat"},
    {"text": "Yeah right, like that's going to happen", "label": 1, "tone": "flat"},
    {"text": "What a surprise, nobody saw that coming", "label": 1, "tone": "exaggerated"},
    {"text": "Oh brilliant, another meeting about meetings", "label": 1, "tone": "flat"},
    {"text": "Thanks so much for that incredibly helpful advice", "label": 1, "tone": "flat"},
    {"text": "Well that went exactly as planned", "label": 1, "tone": "flat"},
    {"text": "Oh how wonderful, more homework", "label": 1, "tone": "exaggerated"},
    {"text": "Really impressive work there buddy", "label": 1, "tone": "flat"},
    {"text": "Oh I just love waiting in long lines", "label": 1, "tone": "flat"},
    {"text": "Wow you really outdid yourself this time", "label": 1, "tone": "exaggerated"},
    {"text": "Perfect timing as always", "label": 1, "tone": "flat"},
    {"text": "Oh sure blame it on the weather", "label": 1, "tone": "flat"},

    # NON-SARCASTIC examples (label = 0)
    {"text": "That was a really great presentation", "label": 0, "tone": "enthusiastic"},
    {"text": "I genuinely enjoyed the movie last night", "label": 0, "tone": "enthusiastic"},
    {"text": "The weather is beautiful today", "label": 0, "tone": "normal"},
    {"text": "Thank you for helping me with this project", "label": 0, "tone": "warm"},
    {"text": "I think we should take a different approach", "label": 0, "tone": "normal"},
    {"text": "Can you pass me the salt please", "label": 0, "tone": "normal"},
    {"text": "The test results look promising", "label": 0, "tone": "enthusiastic"},
    {"text": "I appreciate your honest feedback", "label": 0, "tone": "warm"},
    {"text": "Let me check the schedule for tomorrow", "label": 0, "tone": "normal"},
    {"text": "This is the best pizza I have ever had", "label": 0, "tone": "enthusiastic"},
    {"text": "Good morning, how are you doing", "label": 0, "tone": "warm"},
    {"text": "I need to finish this report by Friday", "label": 0, "tone": "normal"},
    {"text": "That concert was absolutely incredible", "label": 0, "tone": "enthusiastic"},
    {"text": "Could you explain that one more time", "label": 0, "tone": "normal"},
    {"text": "Happy birthday, hope you have a great day", "label": 0, "tone": "warm"},
]


def _generate_synthetic_audio(text: str, tone: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate synthetic audio waveform that mimics speech prosodic patterns.

    Different tones produce different pitch/energy/tempo characteristics:
    - flat: low pitch variation, low energy, slow tempo (sarcastic)
    - exaggerated: high pitch variation, variable energy (sarcastic)
    - enthusiastic: moderate-high pitch, good energy (genuine)
    - normal: moderate pitch, moderate energy (genuine)
    - warm: slightly elevated pitch, smooth energy (genuine)
    """
    duration = 1.5 + len(text) * 0.03  # Longer text = longer audio
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    if tone == "flat":
        # Sarcastic flat tone: low pitch, minimal variation, slow
        base_freq = 110 + np.random.uniform(-10, 10)
        pitch_mod = 3 * np.sin(2 * np.pi * 0.5 * t)  # Very small modulation
        amplitude = 0.15 + 0.02 * np.sin(2 * np.pi * 0.3 * t)  # Low energy
    elif tone == "exaggerated":
        # Sarcastic exaggerated tone: wide pitch swings, variable energy
        base_freq = 180 + np.random.uniform(-20, 20)
        pitch_mod = 60 * np.sin(2 * np.pi * 2.0 * t)  # Large pitch swings
        amplitude = 0.3 + 0.2 * np.sin(2 * np.pi * 1.5 * t)  # Variable energy
    elif tone == "enthusiastic":
        # Genuine enthusiastic: good pitch, strong energy
        base_freq = 160 + np.random.uniform(-15, 15)
        pitch_mod = 25 * np.sin(2 * np.pi * 1.2 * t)
        amplitude = 0.35 + 0.1 * np.sin(2 * np.pi * 1.0 * t)
    elif tone == "warm":
        # Genuine warm: slightly elevated, smooth
        base_freq = 140 + np.random.uniform(-10, 10)
        pitch_mod = 15 * np.sin(2 * np.pi * 0.8 * t)
        amplitude = 0.25 + 0.05 * np.sin(2 * np.pi * 0.6 * t)
    else:  # normal
        base_freq = 130 + np.random.uniform(-10, 10)
        pitch_mod = 12 * np.sin(2 * np.pi * 1.0 * t)
        amplitude = 0.2 + 0.05 * np.sin(2 * np.pi * 0.7 * t)

    # Generate waveform
    freq = base_freq + pitch_mod
    phase = 2 * np.pi * np.cumsum(freq) / sr
    waveform = amplitude * np.sin(phase)

    # Add harmonics for realism
    waveform += 0.3 * amplitude * np.sin(2 * phase)
    waveform += 0.15 * amplitude * np.sin(3 * phase)

    # Add slight noise
    waveform += np.random.normal(0, 0.005, len(waveform))

    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8) * 0.8

    return waveform.astype(np.float32)


def generate_samples(output_dir: str = DATA_DIR, force: bool = False):
    """
    Generate synthetic audio samples and save as .wav files with a manifest.

    Args:
        output_dir: Directory to save samples.
        force: If True, regenerate even if samples exist.
    """
    manifest_path = os.path.join(output_dir, "manifest.json")

    if os.path.exists(manifest_path) and not force:
        print(f"[DataLoader] Samples already exist at {output_dir}. Use force=True to regenerate.")
        return

    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    print(f"[DataLoader] Generating {len(SAMPLE_UTTERANCES)} synthetic samples...")

    for i, sample in enumerate(SAMPLE_UTTERANCES):
        filename = f"sample_{i:03d}.wav"
        filepath = os.path.join(output_dir, filename)

        # Generate and save audio
        waveform = _generate_synthetic_audio(sample["text"], sample["tone"])
        sf.write(filepath, waveform, SAMPLE_RATE)

        manifest.append({
            "id": i,
            "filename": filename,
            "text": sample["text"],
            "label": sample["label"],
            "tone": sample["tone"],
        })

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    sarcastic_count = sum(1 for s in manifest if s["label"] == 1)
    genuine_count = len(manifest) - sarcastic_count
    print(f"[DataLoader] Generated {len(manifest)} samples: {sarcastic_count} sarcastic, {genuine_count} genuine")
    print(f"[DataLoader] Saved to {output_dir}")


class SarcasmDataset(Dataset):
    """PyTorch dataset for sarcasm detection training."""

    def __init__(self, text_embeddings: torch.Tensor, audio_features: torch.Tensor,
                 labels: torch.Tensor):
        self.text_embeddings = text_embeddings
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_embedding": self.text_embeddings[idx],
            "audio_features": self.audio_features[idx],
            "label": self.labels[idx],
        }


def load_dataset(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Load the sample manifest and return list of dicts with text, audio_path, label.
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("[DataLoader] No samples found. Generating...")
        generate_samples(data_dir)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    for item in manifest:
        item["audio_path"] = os.path.join(data_dir, item["filename"])

    return manifest


def create_dataloaders(text_embeddings: torch.Tensor, audio_features: torch.Tensor,
                       labels: torch.Tensor, batch_size: int = BATCH_SIZE):
    """
    Split data into train/val/test and return DataLoaders.
    """
    dataset = SarcasmDataset(text_embeddings, audio_features, labels)

    total = len(dataset)
    train_size = int(TRAIN_SPLIT * total)
    val_size = int(VAL_SPLIT * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"[DataLoader] Split: train={train_size}, val={val_size}, test={test_size}")
    return train_loader, val_loader, test_loader
