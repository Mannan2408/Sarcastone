"""
SarcasTone — Single Prediction Pipeline
==========================================
End-to-end inference: audio → transcription + emoji analysis →
text embedding → audio features + pitch contour →
run all 3 models + semantic-acoustic conflict scoring → return structured analysis.
"""

import os
import torch
import numpy as np
import librosa

from src.config import MODEL_DIR, SAMPLE_RATE, TEXT_EMBEDDING_DIM, AUDIO_FEATURE_DIM
from src.text_encoder import get_text_encoder
from src.audio_prosody import extract_prosody, describe_prosody, extract_pitch_contour
from src.transcriber import get_transcriber
from src.emoji_analyzer import analyze_emojis
from src.models.text_baseline import TextBaseline
from src.models.audio_baseline import AudioBaseline
from src.models.fusion_model import FusionModel
from src.saliency_analyzer import analyze_transcript_saliency


class Predictor:
    """Complete multimodal sarcasm detection pipeline with explainability."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = get_text_encoder()
        self.transcriber = get_transcriber()

        # Load trained models
        self.text_model = self._load_model(TextBaseline(), "text_baseline.pt")
        self.audio_model = self._load_model(AudioBaseline(), "audio_baseline.pt")
        self.fusion_model = self._load_model(FusionModel(), "fusion_model.pt")

        print("[Predictor] All models loaded. Ready for inference.")

    def _load_model(self, model, filename):
        """Load model weights if available, otherwise use untrained model."""
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            print(f"  ✓ Loaded {filename}")
        else:
            print(f"  ⚠️  {filename} not found — using untrained model.")
        model.to(self.device)
        model.eval()
        return model

    def predict(self, audio_path: str = None, audio_array: np.ndarray = None,
                sr: int = SAMPLE_RATE, text_override: str = None) -> dict:
        """
        Run full sarcasm detection pipeline with emoji & pitch contour analysis.

        Args:
            audio_path: Path to .wav file.
            audio_array: Raw audio numpy array.
            sr: Sample rate.
            text_override: If provided, skip transcription and use this text.

        Returns:
            Dict with transcription, predictions, confidence scores,
            emoji semantics, pitch contour series, and conflict meter.
        """
        # Step 1: Load audio
        if audio_array is not None:
            y = audio_array.astype(np.float32)
        elif audio_path is not None:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        else:
            raise ValueError("Either audio_path or audio_array must be provided")

        # Step 2: Transcribe or use override
        if text_override:
            transcription = text_override
        else:
            transcription = self.transcriber.transcribe(audio_array=y, sr=sr)

        # Step 3: Emoji Semantic Analysis
        emoji_info = analyze_emojis(transcription)

        # Use augmented text (where emojis are expanded to semantic tokens) for embedding
        text_for_embedding = emoji_info["augmented_text"] if emoji_info["has_emojis"] else transcription
        text_embedding = self.text_encoder.encode(text_for_embedding)

        # Step 4: Extract audio features & pitch contour curve
        audio_features = extract_prosody(y=y, sr=sr)
        pitch_contour = extract_pitch_contour(y=y, sr=sr, num_points=60)
        prosody_description = describe_prosody(audio_features, pitch_contour)

        # Step 5: Run all 3 models
        text_emb_tensor = text_embedding.unsqueeze(0).to(self.device)
        audio_feat_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            text_logit = self.text_model(text_emb_tensor)
            audio_logit = self.audio_model(audio_feat_tensor)
            fusion_logit = self.fusion_model(text_emb_tensor, audio_feat_tensor)

            text_prob = torch.sigmoid(text_logit).item()
            audio_prob = torch.sigmoid(audio_logit).item()
            fusion_prob = torch.sigmoid(fusion_logit).item()

        # Emoji Modulation: If heavy sarcasm emojis present (e.g. 🙃, 🙄, 🤡), modulate probability
        if emoji_info["has_emojis"]:
            emoji_weight = emoji_info["sarcasm_bias"]
            # Blend model output with emoji signal (65% model, 35% explicit emoji indicator)
            fusion_prob = (fusion_prob * 0.65) + (emoji_weight * 0.35)
            text_prob = (text_prob * 0.60) + (emoji_weight * 0.40)

        is_sarcastic = fusion_prob >= 0.50

        # Step 6: Lexical-Acoustic Saliency Analysis & Token Polarimeter
        pitch_std = float(prosody_description["pitch_std_hz"])
        is_pitch_flat = pitch_std < 25.0
        saliency_data = analyze_transcript_saliency(transcription, pitch_std, is_sarcastic)

        # Step 7: Semantic-Acoustic Conflict Index Calculation
        if emoji_info["has_emojis"] and emoji_info["sarcasm_bias"] > 0.6:
            conflict_index = 82.0 + (14.0 if is_pitch_flat else 6.0)
            conflict_detail = "High discrepancy: Sarcastic emojis contrast with literal wording."
        elif is_sarcastic and is_pitch_flat and (saliency_data["lexical_valence"] > 0.3 or saliency_data["hyperbole_count"] > 0):
            conflict_index = 86.0 + min(11.0, saliency_data["polarity_gap"] * 8.0)
            conflict_detail = f"Contradiction detected: Positive wording ({saliency_data['lexical_valence']:+.2f}) delivered with flat vocal pitch ({pitch_std:.1f} Hz std)."
        elif not is_sarcastic:
            conflict_index = max(6.0, min(30.0, 26.0 - (pitch_std * 0.2)))
            conflict_detail = "Acoustic and lexical signals agree: delivery matches literal meaning."
        else:
            base_diff = abs(text_prob - audio_prob) * 65.0 + (saliency_data["polarity_gap"] * 25.0)
            conflict_index = round(min(96.0, max(12.0, base_diff)), 1)
            conflict_detail = "Cross-modal divergence between text sentiment branch and acoustic prosody branch."

        conflict_index = round(min(98.5, max(5.0, conflict_index)), 1)
        conflict_level = "CRITICAL" if conflict_index > 70 else ("MODERATE" if conflict_index > 38 else "HARMONIC")

        # Step 8: Multi-Factor Sarcasm Radar Vector (6-Axis Polar Analysis)
        flatness_score = float(np.clip(100.0 - (pitch_std * 1.6), 12.0, 98.0))
        hyperbole_score = float(np.clip((saliency_data["hyperbole_count"] * 38.0) + (max(0.0, saliency_data["lexical_valence"]) * 40.0), 10.0, 98.0))
        tempo_val = float(prosody_description.get("tempo_bpm", 120.0))
        drawl_score = float(np.clip(100.0 - (tempo_val * 0.55), 15.0, 95.0))
        rms_val = float(prosody_description.get("energy_rms", 0.05))
        rms_max = float(prosody_description.get("energy_rms_max", rms_val * 1.5))
        contrast_score = float(np.clip((rms_max / max(1e-4, rms_val)) * 24.0, 15.0, 96.0))
        divergence_score = float(np.clip(conflict_index, 10.0, 98.0))
        marker_score = float(np.clip((emoji_info["sarcasm_bias"] * 75.0) + (saliency_data["intensifier_count"] * 28.0), 10.0, 98.0))

        radar_data = {
            "pitch_flatness": round(flatness_score, 1),
            "hyperbolic_intensity": round(hyperbole_score, 1),
            "vocal_drawl": round(drawl_score, 1),
            "acoustic_contrast": round(contrast_score, 1),
            "cross_modal_divergence": round(divergence_score, 1),
            "ironic_marker_load": round(marker_score, 1),
        }

        # Step 9: Delivery Archetype Classification
        if is_sarcastic:
            if is_pitch_flat and saliency_data["lexical_valence"] > 0.2:
                archetype = {
                    "id": "deadpan_delivery",
                    "name": "Deadpan Delivery",
                    "icon": "🧊",
                    "badge": "DEADPAN SARCASM",
                    "tagline": "Flat Pitch with Positive Words",
                    "description": "Positive words spoken with flat, monotone vocal pitch. The absence of pitch inflection signals sarcasm.",
                }
            elif pitch_std > 52.0:
                archetype = {
                    "id": "exaggerated_mockery",
                    "name": "Exaggerated Mockery",
                    "icon": "🎭",
                    "badge": "MOCKING DELIVERY",
                    "tagline": "Theatrical Pitch Swings",
                    "description": "Pronounced pitch swings and elongated syllables used to parody sincere praise.",
                }
            elif emoji_info["has_emojis"] and emoji_info["sarcasm_bias"] > 0.6:
                archetype = {
                    "id": "ironic_contrast",
                    "name": "Ironic Emoji Contrast",
                    "icon": "🙃",
                    "badge": "VISUAL IRONY",
                    "tagline": "Visual Irony Cues",
                    "description": "Sarcastic emojis (such as upside-down face or eye-roll) reverse literal wording.",
                }
            else:
                archetype = {
                    "id": "passive_aggressive",
                    "name": "Passive-Aggressive Tone",
                    "icon": "🫠",
                    "badge": "RESIGNED TONE",
                    "tagline": "Suppressed Energy & Slow Tempo",
                    "description": "Slow tempo and low vocal energy conveying reluctant or cynical compliance.",
                }
        else:
            if saliency_data["lexical_valence"] > 0.2 and pitch_std >= 25.0:
                archetype = {
                    "id": "sincere_praise",
                    "name": "Sincere Praise",
                    "icon": "✨",
                    "badge": "GENUINE EXPRESSION",
                    "tagline": "Natural Pitch Modulation",
                    "description": "Dynamic pitch variation and vocal energy confirm sincere sentiment.",
                }
            else:
                archetype = {
                    "id": "neutral_baseline",
                    "name": "Neutral Baseline",
                    "icon": "⚖️",
                    "badge": "STANDARD CADENCE",
                    "tagline": "Standard Speech Cadence",
                    "description": "Standard conversational pacing without emotional exaggeration or acoustic contradiction.",
                }

        # Step 10: Multimodal Latent Tensor Diagnostics
        text_norm = float(torch.norm(text_emb_tensor).item())
        audio_norm = float(torch.norm(audio_feat_tensor).item())
        total_p = text_prob + audio_prob + 1e-6
        text_influence = round((text_prob / total_p) * 100, 1)
        audio_influence = round((audio_prob / total_p) * 100, 1)
        dominant_modality = "Audio Prosody (Pitch/Energy)" if audio_influence >= text_influence else "Lexical Semantics (DistilBERT)"

        latent_diagnostics = {
            "text_latent_norm": round(text_norm, 2),
            "audio_latent_norm": round(audio_norm, 2),
            "text_dims": TEXT_EMBEDDING_DIM,
            "audio_dims": AUDIO_FEATURE_DIM,
            "fusion_dims": TEXT_EMBEDDING_DIM + AUDIO_FEATURE_DIM,
            "text_influence_pct": text_influence,
            "audio_influence_pct": audio_influence,
            "dominant_modality": dominant_modality,
        }

        # Step 11: Build final structured response
        result = {
            "transcription": transcription,
            "prediction": {
                "label": "SARCASTIC" if is_sarcastic else "NON-SARCASTIC",
                "confidence": round(fusion_prob * 100 if is_sarcastic else (1 - fusion_prob) * 100, 1),
                "sarcasm_probability": round(fusion_prob * 100, 1),
            },
            "model_comparison": {
                "text_only": {
                    "label": "SARCASTIC" if text_prob >= 0.5 else "NON-SARCASTIC",
                    "confidence": round(max(text_prob, 1 - text_prob) * 100, 1),
                    "sarcasm_probability": round(text_prob * 100, 1),
                },
                "audio_only": {
                    "label": "SARCASTIC" if audio_prob >= 0.5 else "NON-SARCASTIC",
                    "confidence": round(max(audio_prob, 1 - audio_prob) * 100, 1),
                    "sarcasm_probability": round(audio_prob * 100, 1),
                },
                "fusion": {
                    "label": "SARCASTIC" if is_sarcastic else "NON-SARCASTIC",
                    "confidence": round(max(fusion_prob, 1 - fusion_prob) * 100, 1),
                    "sarcasm_probability": round(fusion_prob * 100, 1),
                },
            },
            "audio_analysis": prosody_description,
            "pitch_contour": pitch_contour,
            "conflict_analysis": {
                "conflict_index": conflict_index,
                "conflict_level": conflict_level,
                "detail": conflict_detail,
                "pitch_std_hz": round(pitch_std, 1),
                "is_pitch_flat": is_pitch_flat,
            },
            "saliency_analysis": saliency_data,
            "radar_data": radar_data,
            "archetype": archetype,
            "latent_diagnostics": latent_diagnostics,
            "emoji_analysis": emoji_info,
        }

        return result


# Singleton
_predictor_instance = None


def get_predictor() -> Predictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance
