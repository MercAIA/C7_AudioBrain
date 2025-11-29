from dataclasses import dataclass


@dataclass
class AudioBrainConfig:
    # Thresholds for behavior selection
    high_distress: float = 0.6
    low_coherence: float = 0.4
    high_urgency: float = 0.6