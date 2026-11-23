"""
SarcasTone — Emoji Semantic & Sarcasm Analyzer
================================================
Detects emojis in written/transcribed text, assesses sarcasm/irony bias,
and translates emojis into semantic textual tokens so transformer models
(like DistilBERT) don't discard them as [UNK] tokens.
"""

import re
from typing import Dict, List, Tuple

# Comprehensive emoji sarcasm and sentiment mapping
EMOJI_SARCASM_LEXICON: Dict[str, Dict] = {
    # High-probability sarcasm / irony markers
    "🙃": {"name": "upside-down face", "sarcasm_weight": 0.95, "sentiment": "ironic_frustration", "semantic": "sarcastic passive-aggressive smile"},
    "🙄": {"name": "eye roll", "sarcasm_weight": 0.90, "sentiment": "contempt", "semantic": "annoyed sarcastic eye roll"},
    "🤡": {"name": "clown face", "sarcasm_weight": 0.88, "sentiment": "mockery", "semantic": "mocking foolish behavior"},
    "💀": {"name": "skull", "sarcasm_weight": 0.80, "sentiment": "hyperbolic_dread", "semantic": "sarcastic dead reaction"},
    "☠️": {"name": "skull and crossbones", "sarcasm_weight": 0.80, "sentiment": "hyperbolic_dread", "semantic": "sarcastic dead reaction"},
    "😒": {"name": "unamused face", "sarcasm_weight": 0.85, "sentiment": "displeasure", "semantic": "unimpressed skeptical glare"},
    "😏": {"name": "smirking face", "sarcasm_weight": 0.75, "sentiment": "cynical", "semantic": "knowing sarcastic smirk"},
    "🫠": {"name": "melting face", "sarcasm_weight": 0.85, "sentiment": "resigned_irony", "semantic": "forced smile under suffering"},
    "💅": {"name": "nail polish", "sarcasm_weight": 0.75, "sentiment": "snark", "semantic": "dismissive arrogant attitude"},
    "👏": {"name": "clapping hands", "sarcasm_weight": 0.70, "sentiment": "patronizing", "semantic": "slow sarcastic clap"},
    "🧢": {"name": "billed cap", "sarcasm_weight": 0.85, "sentiment": "cynical", "semantic": "calling out a lie or fake statement"},
    "🗑️": {"name": "wastebasket", "sarcasm_weight": 0.80, "sentiment": "contempt", "semantic": "calling it garbage"},
    "😐": {"name": "neutral face", "sarcasm_weight": 0.70, "sentiment": "deadpan", "semantic": "deadpan unimpressed silence"},
    "😑": {"name": "expressionless face", "sarcasm_weight": 0.75, "sentiment": "deadpan", "semantic": "expressionless exasperation"},
    "😂": {"name": "tears of joy", "sarcasm_weight": 0.40, "sentiment": "amused_mockery", "semantic": "laughing reaction"},
    "🤣": {"name": "rofl", "sarcasm_weight": 0.45, "sentiment": "amused_mockery", "semantic": "derisive laughter"},
    "😬": {"name": "grimacing face", "sarcasm_weight": 0.65, "sentiment": "awkward_mockery", "semantic": "awkward cringe"},
    "🤷": {"name": "shrug", "sarcasm_weight": 0.60, "sentiment": "indifference", "semantic": "dismissive shrug"},
    "🤷‍♂️": {"name": "man shrugging", "sarcasm_weight": 0.60, "sentiment": "indifference", "semantic": "dismissive shrug"},
    "🤷‍♀️": {"name": "woman shrugging", "sarcasm_weight": 0.60, "sentiment": "indifference", "semantic": "dismissive shrug"},

    # Genuine / Positive markers (contrast against sarcastic delivery)
    "❤️": {"name": "red heart", "sarcasm_weight": 0.05, "sentiment": "affection", "semantic": "genuine love and warmth"},
    "😊": {"name": "smiling face", "sarcasm_weight": 0.10, "sentiment": "warm", "semantic": "genuine warm smile"},
    "🎉": {"name": "party popper", "sarcasm_weight": 0.15, "sentiment": "celebration", "semantic": "celebration excitement"},
    "🙏": {"name": "folded hands", "sarcasm_weight": 0.10, "sentiment": "gratitude", "semantic": "sincere gratitude"},
    "👍": {"name": "thumbs up", "sarcasm_weight": 0.35, "sentiment": "approval", "semantic": "acknowledgment approval"},
}

# Regex to detect any unicode emojis
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "]+",
    flags=re.UNICODE,
)


def analyze_emojis(text: str) -> Dict:
    """
    Extract and analyze emojis in the given text.

    Returns:
        Dict with:
          - detected: list of {emoji, name, sarcasm_weight, semantic}
          - has_emojis: bool
          - sarcasm_bias: float (0.0 to 1.0)
          - augmented_text: text with emojis replaced by descriptive semantic tokens
    """
    detected = []
    total_weight = 0.0
    augmented = text

    for char, info in EMOJI_SARCASM_LEXICON.items():
        if char in text:
            detected.append({
                "emoji": char,
                "name": info["name"],
                "sarcasm_weight": info["sarcasm_weight"],
                "semantic": info["semantic"],
            })
            total_weight += info["sarcasm_weight"]
            # Augment text for transformer: replace emoji with semantic text token
            augmented = augmented.replace(char, f" [{info['semantic']}] ")

    has_emojis = len(detected) > 0
    avg_sarcasm_bias = (total_weight / len(detected)) if has_emojis else 0.0

    return {
        "detected": detected,
        "has_emojis": has_emojis,
        "sarcasm_bias": round(avg_sarcasm_bias, 3),
        "augmented_text": re.sub(r"\s+", " ", augmented).strip(),
    }
