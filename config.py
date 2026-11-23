"""
SarcasTone Configuration
========================
Central configuration for hyperparameters, paths, and feature dimensions.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Audio Settings ---
SAMPLE_RATE = 16000          # 16kHz standard for speech
MAX_AUDIO_DURATION = 10.0    # Maximum seconds of audio to process
MIN_AUDIO_DURATION = 0.5     # Minimum seconds needed

# --- Prosodic Feature Dimensions ---
# Features extracted by audio_prosody.py:
#   Pitch (F0): mean, std, max, min, range         = 5
#   Energy (RMS): mean, std, max                    = 3
#   Zero Crossing Rate: mean, std                   = 2
#   Spectral Centroid: mean, std                    = 2
#   Tempo                                           = 1
#   MFCCs: 13 coefficients (mean each)              = 13
#   Total                                           = 26
AUDIO_FEATURE_DIM = 26

# --- Text Encoder Settings ---
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_EMBEDDING_DIM = 768     # DistilBERT output dimension

# --- Fusion Model Settings ---
FUSION_HIDDEN_DIM = 256      # Hidden layer size in fusion network
DROPOUT_RATE = 0.3           # Dropout probability
NUM_CLASSES = 1              # Binary classification (sigmoid output)

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_EPOCHS = 30
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Whisper Settings ---
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# --- Flask Settings ---
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
