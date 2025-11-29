# C7_AudioBrain
Implementation of the C7 AudioBrain: a cognitive listener that models intent, emotion, and environment to generate behavior policies before language.
# C7-AudioBrain (C7 Listener)

This repository contains the C7 AudioBrain / Listener implementation.

C7-AudioBrain is a cognitive listener built on top of the C7 Architecture:
it models meaning (A), emotion (B), and environment (C) from
incoming text (and later audio), builds a scene state, and selects a
behavior policy before language.

This is the next layer above the GRSC core engine:
- Full architecture: https://github.com/MercAIA/C7
- Core engine: https://github.com/MercAIA/C7-core

The technical design is documented in:

- docs/C7_Listener_Tech_Note_v1.0.pdf

---

## Goals for v0.1.0

- Input: text only (audio will be added later).
- Output:
  - A/B/C scores (meaning, emotion, environment)
  - Scene state: coherence, distress, urgency, stability
  - A behavior label (e.g. calm_support, clarify, direct_answer, `exit`)

No external APIs are required for this first version.
We focus on the cognitive pipeline, not speech or LLM integration.

---

## Package Layout

- src/c7_audiobrain/features_text.py  
  Basic text feature extraction.

- src/c7_audiobrain/abc_model.py  
  Maps features → A/B/C scores.

- src/c7_audiobrain/scene_state.py  
  Builds the high-level scene (coherence, distress, urgency, stability).

- src/c7_audiobrain/policy.py  
  Selects a behavior label based on the scene.

- src/c7_audiobrain/engine.py  
  Simple pipeline: text → features → A/B/C → scene → behavior.

This repository is the first concrete step toward a full C7 cognitive listener.
