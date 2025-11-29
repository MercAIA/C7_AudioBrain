from dataclasses import asdict
from typing import Any, Dict

from .config import AudioBrainConfig
from .features_text import extract_text_features
from .abc_model import compute_abc
from .scene_state import build_scene
from .policy import select_behavior


class AudioBrainEngine:
    """
    Minimal v0.1 cognitive listener engine.

    Pipeline:
        text -> features -> ABC -> scene -> behavior
    """

    def __init__(self, config: AudioBrainConfig | None = None) -> None:
        self.config = config or AudioBrainConfig()

    def step(self, text: str) -> Dict[str, Any]:
        """
        Process a single text input and return the full cognitive state.
        """
        features = extract_text_features(text)
        abc = compute_abc(features)
        scene = build_scene(abc)
        behavior = select_behavior(scene, self.config)

        return {
            "input": text,
            "features": asdict(features),
            "abc": asdict(abc),
            "scene": asdict(scene),
            "behavior": {
                "label": behavior.label,
                "explanation": behavior.explanation,
            },
        }