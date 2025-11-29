from dataclasses import dataclass
import re
from typing import Dict, List


POSITIVE_WORDS: List[str] = [
    "thanks", "thank you", "great", "good", "love", "happy", "awesome"
]

NEGATIVE_WORDS: List[str] = [
    "angry", "hate", "fuck", "shit", "stupid", "bad", "terrible", "upset",
    "anxious", "anxiety", "depressed", "sad"
]


@dataclass
class TextFeatures:
    length: int
    exclamations: int
    questions: int
    positive_hits: int
    negative_hits: int
    uppercase_ratio: float


def extract_text_features(text: str) -> TextFeatures:
    text_stripped = text.strip()
    length = len(text_stripped)

    exclamations = text_stripped.count("!")
    questions = text_stripped.count("?")

    lowered = text_stripped.lower()

    positive_hits = sum(1 for w in POSITIVE_WORDS if w in lowered)
    negative_hits = sum(1 for w in NEGATIVE_WORDS if w in lowered)

    if length > 0:
        uppercase_chars = len(re.findall(r"[A-Z]", text_stripped))
        uppercase_ratio = uppercase_chars / length
    else:
        uppercase_ratio = 0.0

    return TextFeatures(
        length=length,
        exclamations=exclamations,
        questions=questions,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        uppercase_ratio=uppercase_ratio,
    )