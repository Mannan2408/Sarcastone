"""
SarcasTone — Audio Prosody Extractor
=====================================
Extracts prosodic and acoustic features from speech audio using Librosa.

Features extracted (26 total):
  - Pitch (F0): mean, std, max, min, range          [5]
  - Energy (RMS): mean, std, max                     [3]
  - Zero Crossing Rate: mean, std                    [2]
  - Spectral Centroid: mean, std                     [2]
  - Tempo (beats per minute)                         [1]
  - MFCCs: 13 coefficients (mean of each)            [13]
"""

import numpy as np
import librosa
from src.config import SAMPLE_RATE, AUDIO_FEATURE_DIM, MAX_AUDIO_DURATION


def extract_prosody(audio_path: str = None, y: np.ndarray = None, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract prosodic features from an audio file or raw waveform.

    Args:
        audio_path: Path to a .wav audio file (used if y is None).
        y: Raw audio waveform as numpy array (optional, overrides audio_path).
        sr: Sample rate (default 16000).

    Returns:
        1D numpy array of shape (26,) with all prosodic features.
    """
    # Load audio if waveform not provided
    if y is None:
        if audio_path is None:
            raise ValueError("Either audio_path or y must be provided")
        y, sr = librosa.load(audio_path, sr=sr, duration=MAX_AUDIO_DURATION)

    # Ensure minimum length
    if len(y) < int(sr * 0.1):
        # Pad very short audio
        y = np.pad(y, (0, int(sr * 0.5) - len(y)), mode='constant')

    features = []

    # --- 1. Pitch (F0) using pyin ---
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0_valid = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
    features.extend([
        np.mean(f0_valid),       # F0 mean
        np.std(f0_valid),        # F0 std (pitch variation)
        np.max(f0_valid),        # F0 max
        np.min(f0_valid),        # F0 min
        np.ptp(f0_valid),        # F0 range (max - min)
    ])

    # --- 2. Energy (RMS) ---
    rms = librosa.feature.rms(y=y)[0]
    features.extend([
        np.mean(rms),            # RMS mean (average loudness)
        np.std(rms),             # RMS std (loudness variation)
        np.max(rms),             # RMS max (peak loudness)
    ])

    # --- 3. Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features.extend([
        np.mean(zcr),            # ZCR mean
        np.std(zcr),             # ZCR std
    ])

    # --- 4. Spectral Centroid ---
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend([
        np.mean(sc),             # Spectral centroid mean
        np.std(sc),              # Spectral centroid std
    ])

    # --- 5. Tempo ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 0.0
    features.append(tempo_val)

    # --- 6. MFCCs (13 coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)  # Mean of each coefficient
    features.extend(mfcc_means.tolist())

    feature_vector = np.array(features, dtype=np.float32)

    # Replace any NaN or Inf with 0
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    assert feature_vector.shape[0] == AUDIO_FEATURE_DIM, \
        f"Expected {AUDIO_FEATURE_DIM} features, got {feature_vector.shape[0]}"

    return feature_vector


def get_feature_names() -> list[str]:
    """Return human-readable names for each feature dimension."""
    return [
        "Pitch (F0) Mean", "Pitch (F0) Std", "Pitch (F0) Max",
        "Pitch (F0) Min", "Pitch (F0) Range",
        "Energy (RMS) Mean", "Energy (RMS) Std", "Energy (RMS) Max",
        "Zero Crossing Rate Mean", "Zero Crossing Rate Std",
        "Spectral Centroid Mean", "Spectral Centroid Std",
        "Tempo (BPM)",
    ] + [f"MFCC {i+1}" for i in range(13)]


def extract_pitch_contour(y: np.ndarray, sr: int = SAMPLE_RATE, num_points: int = 60) -> list[dict]:
    """
    Extract a normalized F0 pitch contour time series for UI rendering.

    Returns:
        List of dicts: [{"t": round(time, 2), "hz": round(freq, 1), "voiced": bool}]
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        times = librosa.times_like(f0, sr=sr)
        
        # Sample down to num_points uniformly
        if len(times) > num_points:
            indices = np.linspace(0, len(times) - 1, num_points, dtype=int)
            times = times[indices]
            f0 = f0[indices]
            voiced_flag = voiced_flag[indices]

        contour = []
        for t, freq, voiced in zip(times, f0, voiced_flag):
            val = float(freq) if not np.isnan(freq) and voiced else 0.0
            contour.append({
                "t": round(float(t), 2),
                "hz": round(val, 1),
                "voiced": bool(voiced and val > 0),
            })
        return contour
    except Exception as e:
        print(f"[AudioProsody] Contour error: {e}")
        return []


def describe_prosody(features: np.ndarray, contour: list[dict] = None) -> dict:
    """
    Generate human-readable descriptions of the prosodic features
    for display in the UI.

    Returns:
        Dict with keys: pitch, energy, pace, variation — each with
        a label and a short description.
    """
    names = get_feature_names()
    f = {name: val for name, val in zip(names, features)}

    # Pitch assessment
    pitch_std = f["Pitch (F0) Std"]
    if pitch_std < 20:
        pitch_desc = {"label": "Flat", "detail": "Very low pitch variation — monotone delivery", "icon": "📉"}
    elif pitch_std < 50:
        pitch_desc = {"label": "Normal", "detail": "Moderate pitch variation — natural speech", "icon": "📊"}
    else:
        pitch_desc = {"label": "Exaggerated", "detail": "High pitch variation — possibly theatrical or mocking", "icon": "📈"}

    # Energy assessment
    rms_mean = f["Energy (RMS) Mean"]
    if rms_mean < 0.02:
        energy_desc = {"label": "Low", "detail": "Quiet or mumbled delivery", "icon": "🔈"}
    elif rms_mean < 0.08:
        energy_desc = {"label": "Normal", "detail": "Standard speaking volume", "icon": "🔊"}
    else:
        energy_desc = {"label": "High", "detail": "Loud or emphatic delivery", "icon": "🔊"}

    # Pace assessment
    tempo = f["Tempo (BPM)"]
    if tempo < 80:
        pace_desc = {"label": "Slow", "detail": "Unusually slow — possibly deliberate or mocking", "icon": "🐢"}
    elif tempo < 160:
        pace_desc = {"label": "Normal", "detail": "Standard speaking pace", "icon": "⏱️"}
    else:
        pace_desc = {"label": "Fast", "detail": "Rapid delivery — excited or rushing", "icon": "⚡"}

    # MFCC band descriptions and normalized power (0-100)
    mfcc_labels = [
        "Band 1: Energy & Power",
        "Band 2: Low-Vocal F1 Formant",
        "Band 3: Mid-Vowel F2 Formant",
        "Band 4: Upper-Formant F3 Formant",
        "Band 5: Nasal/Vocal Tilt",
        "Band 6: Lower Consonant Articulation",
        "Band 7: Mid-Consonant Sibilance",
        "Band 8: Upper Resonance",
        "Band 9: Glottal Air Flow",
        "Band 10: Speech Texture",
        "Band 11: Treble Resonance",
        "Band 12: High-Frequency Presence",
        "Band 13: Spectral Air & Breath",
    ]
    mfcc_bands = []
    raw_mfcc_vals = [float(f[f"MFCC {i+1}"]) for i in range(13)]
    min_m = min(raw_mfcc_vals) if raw_mfcc_vals else 0
    max_m = max(raw_mfcc_vals) if raw_mfcc_vals else 1
    m_range = max(1e-5, max_m - min_m)

    for i in range(13):
        val = raw_mfcc_vals[i]
        norm_pct = int(np.clip(((val - min_m) / m_range) * 100, 5, 98))
        mfcc_bands.append({
            "index": i + 1,
            "band": f"MFCC {i+1}",
            "label": mfcc_labels[i],
            "val": round(val, 2),
            "power_pct": norm_pct,
        })

    return {
        "pitch": pitch_desc,
        "energy": energy_desc,
        "pace": pace_desc,
        "pitch_mean_hz": float(f["Pitch (F0) Mean"]),
        "pitch_std_hz": float(pitch_std),
        "pitch_min_hz": float(f["Pitch (F0) Min"]),
        "pitch_max_hz": float(f["Pitch (F0) Max"]),
        "pitch_range_hz": float(f["Pitch (F0) Range"]),
        "energy_rms": float(rms_mean),
        "energy_rms_max": float(f["Energy (RMS) Max"]),
        "energy_rms_std": float(f["Energy (RMS) Std"]),
        "tempo_bpm": float(tempo),
        "spectral_centroid_mean": float(f["Spectral Centroid Mean"]),
        "spectral_centroid_std": float(f["Spectral Centroid Std"]),
        "zero_crossing_rate_mean": float(f["Zero Crossing Rate Mean"]),
        "zero_crossing_rate_std": float(f["Zero Crossing Rate Std"]),
        "mfcc_bands": mfcc_bands,
        "contour": contour or [],
    }
