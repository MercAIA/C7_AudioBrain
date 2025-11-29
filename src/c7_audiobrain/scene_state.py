from dataclasses import dataclass
from .abc_model import ABCState


@dataclass
class SceneState:
    coherence: float
    distress: float
    urgency: float
    stability: float


def build_scene(abc: ABCState) -> SceneState:
    # coherence: when meaning is higher and emotion is not too chaotic
    coherence = max(0.0, abc.A_meaning - 0.3 * abc.B_emotion)

    # distress: mostly emotional intensity plus environment
    distress = min(1.0, 0.7 * abc.B_emotion + 0.4 * abc.C_environment)

    # urgency: mixture of distress and environment pressure
    urgency = min(1.0, 0.6 * distress + 0.6 * abc.C_environment)

    # stability: inverse of distress, but modulated by coherence
    stability = max(0.0, (1.0 - distress) * 0.7 + 0.3 * coherence)

    return SceneState(
        coherence=coherence,
        distress=distress,
        urgency=urgency,
        stability=stability,
    )