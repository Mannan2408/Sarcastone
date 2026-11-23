"""
SarcasTone — Lexical-Acoustic Saliency & Token Polarimeter
==========================================================
Analyzes lexical sentiment, detects hyperbole/superlative praise words,
and computes word-level saliency and cross-modal polarity inversion.
"""

import re
from typing import Dict, List, Any

# Lexicon of superlative praise words (textbook sarcasm targets)
PRAISE_SUPERLATIVES = {
    "amazing": 0.95, "awesome": 0.90, "brilliant": 0.95, "excellent": 0.90,
    "fantastic": 0.95, "fabulous": 0.90, "genius": 0.98, "great": 0.85,
    "incredible": 0.95, "masterclass": 0.95, "marvelous": 0.90, "perfect": 0.98,
    "perfectly": 0.95, "outstanding": 0.92, "phenomenal": 0.98, "splendid": 0.88,
    "stellar": 0.92, "stunning": 0.90, "superb": 0.92, "terrific": 0.88,
    "wonderful": 0.92, "love": 0.85, "adore": 0.88, "best": 0.95,
    "impressed": 0.80, "exciting": 0.82, "thrilling": 0.85, "flawless": 0.98,
    "beautiful": 0.82, "miracle": 0.95, "hero": 0.85, "champion": 0.88,
}

# Ironic intensifiers & discourse cues that signal sarcasm
IRONY_INTENSIFIERS = {
    "really": 0.70, "totally": 0.75, "absolutely": 0.80, "obviously": 0.85,
    "clearly": 0.80, "definitely": 0.75, "sure": 0.65, "surely": 0.70,
    "yeah": 0.60, "right": 0.60, "wow": 0.75, "oh": 0.50,
    "thanks": 0.60, "thank": 0.55, "nobody": 0.70, "ever": 0.60,
    "completely": 0.70, "certainly": 0.70, "truly": 0.65, "naturally": 0.75,
    "typical": 0.80, "classic": 0.75, "surprise": 0.70,
}

# Negative sentiment words
NEGATIVE_WORDS = {
    "awful": -0.90, "bad": -0.70, "terrible": -0.95, "horrible": -0.95,
    "worst": -0.98, "fail": -0.85, "failure": -0.90, "disaster": -0.95,
    "broken": -0.75, "garbage": -0.90, "rubbish": -0.85, "useless": -0.90,
    "hate": -0.90, "annoying": -0.75, "pathetic": -0.92, "boring": -0.70,
    "ruined": -0.85, "trash": -0.90, "crap": -0.85, "stupid": -0.85,
    "mess": -0.75, "nightmare": -0.92,
}


def analyze_transcript_saliency(transcript: str, pitch_std_hz: float, is_sarcastic_pred: bool) -> Dict[str, Any]:
    """
    Perform token-level saliency and polarity breakdown on speech transcript.

    Args:
        transcript: Transcribed speech text.
        pitch_std_hz: Acoustic pitch variation in Hz.
        is_sarcastic_pred: Whether the model classified the utterance as sarcastic.

    Returns:
        Structured dictionary containing word tokens, polarity metrics,
        and cross-modal inversion indices.
    """
    if not transcript or not transcript.strip():
        return {
            "tokens": [],
            "lexical_valence": 0.0,
            "acoustic_valence": 0.0,
            "polarity_gap": 0.0,
            "hyperbole_count": 0,
            "intensifier_count": 0,
            "narrative": "No speech transcript available."
        }

    raw_tokens = re.findall(r"[\w']+|[.,!?;:]|[^\s\w]", transcript)
    analyzed_tokens: List[Dict[str, Any]] = []

    total_valence = 0.0
    scored_word_count = 0
    hyperbole_count = 0
    intensifier_count = 0

    # Determine acoustic valence based on pitch modulation and prediction
    if is_sarcastic_pred:
        if pitch_std_hz < 10.0:
            acoustic_valence = -0.90
        elif pitch_std_hz < 25.0:
            acoustic_valence = -0.75
        elif pitch_std_hz > 55.0:
            acoustic_valence = -0.65
        else:
            acoustic_valence = -0.45
    else:
        if pitch_std_hz >= 25.0:
            acoustic_valence = +0.85
        else:
            acoustic_valence = +0.60

    for tok in raw_tokens:
        clean = tok.lower().strip(".,!?;:\"'")
        category = "structural"
        valence = 0.0
        sarcasm_saliency = 0.05
        is_highlight = False

        if clean in PRAISE_SUPERLATIVES:
            category = "praise"
            valence = PRAISE_SUPERLATIVES[clean]
            hyperbole_count += 1
            # If spoken in a flat/cynical voice, positive praise is the prime sarcasm trigger!
            sarcasm_saliency = 0.95 if (pitch_std_hz < 25.0 or is_sarcastic_pred) else 0.15
            is_highlight = True
            total_valence += valence
            scored_word_count += 1

        elif clean in IRONY_INTENSIFIERS:
            category = "intensifier"
            valence = 0.45
            intensifier_count += 1
            sarcasm_saliency = 0.80 if is_sarcastic_pred else 0.20
            is_highlight = True
            total_valence += valence
            scored_word_count += 1

        elif clean in NEGATIVE_WORDS:
            category = "negative"
            valence = NEGATIVE_WORDS[clean]
            sarcasm_saliency = 0.40
            is_highlight = True
            total_valence += valence
            scored_word_count += 1

        elif len(clean) > 2 and clean.isalpha():
            # Neutral content word
            category = "content"
            valence = 0.05
            sarcasm_saliency = 0.15
        elif re.match(r"[.,!?;:]", tok):
            category = "punctuation"
            valence = 0.0
            sarcasm_saliency = 0.10

        analyzed_tokens.append({
            "token": tok,
            "category": category,
            "lexical_valence": round(valence, 2),
            "sarcasm_saliency": round(sarcasm_saliency, 2),
            "is_highlight": is_highlight,
            "acoustic_alignment": "CONTRADICTION" if (valence > 0.5 and acoustic_valence < 0) else "HARMONIC",
        })

    avg_lexical_valence = (total_valence / max(1, scored_word_count)) if scored_word_count > 0 else 0.10
    avg_lexical_valence = max(-1.0, min(1.0, round(avg_lexical_valence, 2)))

    # Polarity Inversion Gap: Distance between literal word sentiment and vocal tone
    polarity_gap = round(abs(avg_lexical_valence - acoustic_valence), 2)

    # Explanatory narrative
    if avg_lexical_valence > 0.4 and acoustic_valence < -0.4:
        narrative = f"Positive words (lexical score {avg_lexical_valence:+.2f}) delivered with flat, monotone pitch ({acoustic_valence:+.2f}), indicating sarcastic contrast."
    elif avg_lexical_valence > 0.4 and acoustic_valence > 0.4:
        narrative = f"Positive words ({avg_lexical_valence:+.2f}) align with varied vocal pitch ({acoustic_valence:+.2f}), reflecting sincere praise."
    elif is_sarcastic_pred:
        narrative = f"Vocal tone diverges from literal wording (contrast delta: {polarity_gap:.2f})."
    else:
        narrative = f"Speech prosody ({acoustic_valence:+.2f}) and wording ({avg_lexical_valence:+.2f}) match standard conversational tone."

    return {
        "tokens": analyzed_tokens,
        "lexical_valence": avg_lexical_valence,
        "acoustic_valence": round(acoustic_valence, 2),
        "polarity_gap": polarity_gap,
        "hyperbole_count": hyperbole_count,
        "intensifier_count": intensifier_count,
        "narrative": narrative,
    }
