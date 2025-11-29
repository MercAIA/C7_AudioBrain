from dataclasses import dataclass
from typing import Dict

from .features_text import TextFeatures


@dataclass
class ABCState:
    A_meaning: float   # how contentful / focused the message is
    B_emotion: float   # emotional intensity (0 calm, 1 intense)
    C_environment: float  # external pressure / context intensity


def clamp(x: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(max_v, x))


def compute_abc(features: TextFeatures) -> ABCState:
    # crude heuristics for v0.1, just to get a working pipeline

    # A: more length and more question marks = more meaning/focus
    a_raw = 0.3 + 0.0007 * features.length + 0.15 * min(features.questions, 3)
    A = clamp(a_raw)

    # B: emotional intensity from exclamations, negative words, uppercase
    b_raw = (
        0.15 * min(features.exclamations, 3)
        + 0.1 * min(features.positive_hits + features.negative_hits, 4)
        + 1.5 * features.uppercase_ratio
    )
    B = clamp(b_raw)

    # C: environment pressure approx from questions + negative hits
    c_raw = 0.1 * min(features.questions, 4) + 0.15 * min(features.negative_hits, 4)
    C = clamp(c_raw)

    return ABCState(A_meaning=A, B_emotion=B, C_environment=C)