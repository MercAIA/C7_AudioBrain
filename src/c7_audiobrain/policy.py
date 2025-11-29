from dataclasses import dataclass
from .scene_state import SceneState
from .config import AudioBrainConfig


@dataclass
class BehaviorDecision:
    label: str
    explanation: str


def select_behavior(scene: SceneState, cfg: AudioBrainConfig) -> BehaviorDecision:
    # Very simple v0.1 behavior policy

    if scene.distress >= cfg.high_distress:
        return BehaviorDecision(
            label="calm_support",
            explanation="User seems distressed; respond with calming, supportive tone.",
        )

    if scene.coherence <= cfg.low_coherence:
        return BehaviorDecision(
            label="clarify",
            explanation="Message is low coherence; ask clarifying questions.",
        )

    if scene.urgency >= cfg.high_urgency:
        return BehaviorDecision(
            label="direct_answer",
            explanation="User seems urgent; provide a short, direct answer.",
        )

    if scene.stability < 0.3:
        return BehaviorDecision(
            label="pause_or_exit",
            explanation="Low stability; consider pausing or gently closing the interaction.",
        )

    return BehaviorDecision(
        label="normal_dialogue",
        explanation="Situation is stable; proceed with normal conversational behavior.",
    )